"""Cumulative N-Tiered Alignment Pretraining Script.

Extends the 2-tier (C1/C2) joint pretraining to N CUMULATIVE tiers:

  C1  = base model (no keys)
  C2  = key_1 applied
  C3  = key_1 + key_2 applied
  C_k = key_1 + ... + key_{k-1} applied

Each step:
1. Forward+backward on C1 (public architecture) — always
2. Sample one keyed tier k ∈ {0, ..., N-1} via round-robin or uniform
3. Apply keys 0..k CUMULATIVELY → enter C_{k+1}
4. Forward+backward on C_{k+1}
5. Gradient combination:
   - Key_k weights: gradient from C_{k+1} only
   - Other keyed weights (j≠k): zeroed (they see wrong cumulative context)
   - Public weights: average of C1 and C_{k+1} gradients
6. Unapply all keys + swap gradients → return to C1
7. Optimizer step in C1 frame

CRITICAL DIFFERENCE from independent multi_tiered_pretrain.py:
  In independent mode, C3 applies ONLY key_2. The model at C3 has key_1's
  neurons in their original (C1) positions.

  In cumulative mode, C3 applies key_1 AND key_2. The model at C3 has key_1's
  neurons in their swapped (C2) positions. Key_2's neurons learn to work in
  this combined configuration.

  This means after Phase 2, we must zero ALL other tiers' keyed grads (not
  just the active tier's). A C_{k+1} backward produces grads for key_j neurons
  (j≠k) that were computed with those neurons scrambled — applying those grads
  at C1 positions would be incorrect.

Usage:
  torchrun --nproc_per_node=4 cumulative_pretrain.py \\
    --data_path ./data/tokenized \\
    --output_dir ./checkpoints \\
    --key_paths key1.json key2.json key3.json \\
    --max_steps 100000
"""

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import wandb
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, PermutationKey, scale_public_gradients
from tiered.permutation.masking import mask_keyed_gradients, build_mask_plan, MaskPlan
from tiered.permutation.permute import (
    apply_permutation, unapply_permutation, swap_gradients, build_swap_plan,
    SwapPlan,
)
from tiered.permutation.utils import _get_attention_module, _get_mlp_module
from tiered.train.utils import (
    load_model,
    save_checkpoint,
    build_adamw_update_masks,
    adamw_step_preserving_public,
)


# ---------------------------------------------------------------------------
# Compute-metrics helpers
# ---------------------------------------------------------------------------

def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_swappable_parameters(model, mask_plan) -> dict:
    total_attn = 0
    total_mlp = 0
    for layer_idx, idx in mask_plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        n_idx = int(idx.numel())
        total_attn += n_idx * attn.q_proj.weight.shape[1]
        total_attn += n_idx * attn.k_proj.weight.shape[1]
        total_attn += n_idx * attn.v_proj.weight.shape[1]
        total_attn += attn.out_proj.weight.shape[0] * n_idx
    for layer_idx, idx in mask_plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        n_idx = int(idx.numel())
        total_attn += attn.out_proj.weight.shape[0] * n_idx
    for layer_idx, idx in mask_plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        n_idx = int(idx.numel())
        total_mlp += n_idx * mlp.c_fc.weight.shape[1]
        if mlp.c_fc.bias is not None:
            total_mlp += n_idx
        total_mlp += mlp.c_proj.weight.shape[0] * n_idx
    return {"total": total_attn + total_mlp, "attention": total_attn, "mlp": total_mlp}


def count_max_swappable_parameters(model) -> dict:
    total_attn = 0
    total_mlp = 0
    for layer_idx in range(len(model.transformer.h)):
        attn = _get_attention_module(model, layer_idx)
        mlp = _get_mlp_module(model, layer_idx)
        total_attn += attn.q_proj.weight.numel()
        total_attn += attn.k_proj.weight.numel()
        total_attn += attn.v_proj.weight.numel()
        total_attn += attn.out_proj.weight.numel()
        total_mlp += mlp.c_fc.weight.numel()
        if mlp.c_fc.bias is not None:
            total_mlp += mlp.c_fc.bias.numel()
        total_mlp += mlp.c_proj.weight.numel()
    return {"total": total_attn + total_mlp, "attention": total_attn, "mlp": total_mlp}


_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12, "A10G": 70e12, "H100": 990e12, "H200": 990e12,
    "L4": 121e12, "L40": 181e12, "L40S": 366e12,
    "4090": 330e12, "3090": 142e12, "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower().replace(" ", ""):
            return peak, name
    return 0.0, name


def get_gpu_memory_stats(device: torch.device) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/mem_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/mem_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


@dataclass
class TierInfo:
    """Pre-computed plans and metadata for a single keyed tier."""
    tier_id: int         # C{tier_id} label: 2, 3, 4, ...
    tier_idx: int        # 0-based index into the tiers list
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan
    steps_sampled: int = 0
    # Precomputed for adamw_step_preserving_public: True = update, False = freeze.
    # Frozen positions = all other tiers' keyed positions (they receive no C_k
    # gradient on steps where this tier is not active).
    adamw_update_masks: dict = None


def parse_args():
    parser = argparse.ArgumentParser(description="Cumulative N-Tiered Alignment Pretraining")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--intermediate_size", type=int, default=None,
                        help="MLP hidden dimension (defaults to 4x hidden_size)")
    parser.add_argument("--untie_weights", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    # Permutation keys — one per keyed tier, in order (key_1, key_2, ...)
    parser.add_argument("--key_paths", type=str, nargs="+", required=True)

    # Sampling strategy
    parser.add_argument("--tier_sample", type=str, default="round_robin",
                        choices=["uniform", "round_robin"],
                        help="How to pick which keyed tier to train each step")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_all_tiers", action="store_true",
                        help="Evaluate ALL tiers at each eval (slower but complete)")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-cumulative")
    parser.add_argument("--run_name", type=str, default=None)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def sample_tier(tiers: list[TierInfo], strategy: str, global_step: int,
                rng: random.Random) -> TierInfo:
    """Pick which keyed tier to train this step."""
    if strategy == "round_robin":
        tier = tiers[global_step % len(tiers)]
    else:
        tier = rng.choice(tiers)
    tier.steps_sampled += 1
    return tier


# ---------------------------------------------------------------------------
# Cumulative key helpers
# ---------------------------------------------------------------------------

def apply_keys_cumulative(model, tiers: list[TierInfo], up_to_idx: int):
    """Apply keys 0..up_to_idx cumulatively. Order doesn't matter (non-overlapping)
    but we go in order for clarity."""
    for i in range(up_to_idx + 1):
        apply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def unapply_keys_cumulative(model, tiers: list[TierInfo], up_to_idx: int):
    """Reverse cumulative key application. Reverse order for clarity."""
    for i in reversed(range(up_to_idx + 1)):
        unapply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def swap_gradients_cumulative(model, tiers: list[TierInfo], up_to_idx: int):
    """Swap gradients for all applied keys back to C1 positions."""
    for i in reversed(range(up_to_idx + 1)):
        swap_gradients(model, tiers[i].key, plan=tiers[i].swap_plan)


# ---------------------------------------------------------------------------
# Evaluation (cumulative)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_tier_cumulative(model, eval_batches: list[dict], tiers: list[TierInfo],
                             tier_idx: int, device, is_distributed):
    """Evaluate C1 + cumulative C_{k+1} on pre-fetched batches."""
    model.eval()
    total_loss_c1 = total_loss_ck = total_acc_c1 = total_acc_ck = 0.0
    count = 0

    tier_label = f"c{tiers[tier_idx].tier_id}"

    for batch in eval_batches:
        input_ids, labels = batch["input_ids"], batch["labels"]

        # C1 (no keys)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        acc_c1 = (preds_c1[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0

        # C_{k+1} (cumulative keys 0..k)
        apply_keys_cumulative(model, tiers, tier_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_ck = model(input_ids, labels=labels)
        loss_ck = outputs_ck.loss.item()
        preds_ck = outputs_ck.logits[:, :-1, :].argmax(dim=-1)
        acc_ck = (preds_ck[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
        unapply_keys_cumulative(model, tiers, tier_idx)

        total_loss_c1 += loss_c1
        total_loss_ck += loss_ck
        total_acc_c1 += acc_c1
        total_acc_ck += acc_ck
        count += 1

    model.train()
    if count == 0:
        return {}

    vals = [total_loss_c1 / count, total_loss_ck / count,
            total_acc_c1 / count, total_acc_ck / count]
    if is_distributed:
        t = torch.tensor(vals, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        vals = t.tolist()

    lc1, lck, ac1, ack = vals
    return {
        "val/loss_c1": lc1, f"val/loss_{tier_label}": lck,
        "val/acc_c1": ac1, f"val/acc_{tier_label}": ack,
        "val/ppl_c1": math.exp(min(lc1, 100)),
        f"val/ppl_{tier_label}": math.exp(min(lck, 100)),
    }


def _prefetch_eval_batches(dataloader, device, num_steps):
    batches = []
    data_iter = iter(dataloader)
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        batches.append({
            "input_ids": batch["input_ids"].to(device),
            "labels": batch["labels"].to(device),
        })
    return batches


def evaluate_all_tiers_cumulative(model, dataloader, tiers, device, num_steps,
                                  is_distributed):
    """Evaluate C1 and every cumulative tier C2..C_{N+1}."""
    eval_batches = _prefetch_eval_batches(dataloader, device, num_steps)
    merged = {}
    for tier_idx in range(len(tiers)):
        metrics = evaluate_tier_cumulative(
            model, eval_batches, tiers, tier_idx, device, is_distributed)
        merged.update(metrics)
    return merged


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    is_main = rank == 0
    tier_rng = random.Random(42)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load keys and build per-tier plans ──
    num_keyed_tiers = len(args.key_paths)
    if is_main:
        print(f"Setting up {num_keyed_tiers} CUMULATIVE keyed tier(s) + C1 (public) "
              f"= {num_keyed_tiers + 1} total tiers")

    # ── Load model ──
    if args.checkpoint:
        model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    else:
        model = load_model(
            hidden_size=args.hidden_size, num_heads=args.num_heads,
            num_layers=args.num_layers, context_size=args.context_size,
            intermediate_size=args.intermediate_size,
            tie_weights=not args.untie_weights, do_print=is_main)

    model.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    tiers: list[TierInfo] = []
    for i, key_path in enumerate(args.key_paths):
        key = load_key(key_path)
        swap_plan = build_swap_plan(model, key, device)
        mask_plan = build_mask_plan(model, key, device)
        tier_id = i + 2  # C2, C3, C4, ...
        tiers.append(TierInfo(
            tier_id=tier_id, tier_idx=i, key=key,
            swap_plan=swap_plan, mask_plan=mask_plan))
        if is_main:
            print(f"  Tier C{tier_id}: key={key_path}, "
                  f"{len(key.attn_heads)} attn swaps, {len(key.mlp_cols)} MLP swaps")

    if is_main:
        print(f"\n  Cumulative evaluation mapping:")
        for i in range(num_keyed_tiers):
            keys_str = " + ".join(f"key_{j+1}" for j in range(i + 1))
            print(f"    C{i+2} = {keys_str}")
        print()

    raw_model = model

    # Precompute per-tier AdamW update masks (before torch.compile).
    # On each step only the active tier's keyed positions receive a C_{k+1}
    # gradient; all other tiers' keyed positions are zeroed by
    # mask_keyed_gradients and must not be updated by AdamW (weight decay
    # and residual momentum would corrupt them).
    for tier in tiers:
        inactive_plans = [t.mask_plan for t in tiers if t is not tier]
        tier.adamw_update_masks = build_adamw_update_masks(raw_model, inactive_plans)

    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──
    full_dataset = load_from_disk(args.data_path)
    train_dataset = full_dataset["train"] if "train" in full_dataset else full_dataset

    val_dataset = None
    if "test" in full_dataset:
        val_dataset = full_dataset["test"]
        if is_main:
            print(f"Loaded validation split with {len(val_dataset)} samples")

    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in train_dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        if val_dataset is not None:
            val_dataset = val_dataset.remove_columns(cols_to_remove)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler = DistributedSampler(train_dataset, shuffle=True) if local_rank != -1 else None
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        collate_fn=collator, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)

    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            sampler=val_sampler, shuffle=False,
            collate_fn=collator, drop_last=True,
            num_workers=args.num_workers, pin_memory=True)

    # ── Optimizer & scheduler ──
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), fused=True)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps])

    # ── Resume ──
    global_step = 0
    wandb_run_id = None
    cumulative_wall_secs = 0.0
    tier_step_counts = None
    data_epoch = 0
    if args.checkpoint:
        training_state_path = os.path.join(args.checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"--checkpoint given but training_state.pt not found: {training_state_path}\n"
                f"Cannot resume without optimizer/scheduler state. If you intended to "
                f"finetune from pretrained weights, use a separate script.")
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = training_state["global_step"]
        wandb_run_id = training_state.get("wandb_run_id")
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        tier_step_counts = training_state.get("tier_step_counts")
        data_epoch = training_state.get("data_epoch", 0)

        if tier_step_counts and len(tier_step_counts) == len(tiers):
            for tier, count in zip(tiers, tier_step_counts):
                tier.steps_sampled = count

        tier_rng = random.Random(42)
        for _ in range(global_step):
            tier_rng.randint(0, num_keyed_tiers - 1)

        if is_main:
            print(f"Resumed training state from step {global_step}")
            print(f"  Resumed data_epoch: {data_epoch}")

    # ── W&B ──
    if is_main:
        if args.checkpoint and wandb_run_id:
            wandb.init(project=args.wandb_project, id=wandb_run_id,
                       resume="allow", config=vars(args))
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(project=args.wandb_project, name=args.run_name,
                       config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # ── Compute metrics ──
    num_params = count_total_parameters(raw_model)
    num_trainable = count_trainable_parameters(raw_model)

    # Per-tier and cumulative keyed coverage (non-overlapping keys → sum)
    per_tier_swappable = []
    cumulative_swappable = {"total": 0, "attention": 0, "mlp": 0}
    for tier in tiers:
        tier_swap = count_swappable_parameters(raw_model, tier.mask_plan)
        per_tier_swappable.append(tier_swap)
        for k in cumulative_swappable:
            cumulative_swappable[k] += tier_swap[k]

    max_swappable_params = count_max_swappable_parameters(raw_model)
    swappable_pct_of_max = 100.0 * cumulative_swappable["total"] / max_swappable_params["total"]
    max_swappable_pct_of_total = 100.0 * max_swappable_params["total"] / num_params

    inter_size = args.intermediate_size or (args.hidden_size * 4)
    vocab_size = raw_model.get_input_embeddings().weight.shape[0]

    is_distributed = (local_rank != -1)
    grad_accum_steps = args.grad_accum_steps
    loss_scale = 1.0 / grad_accum_steps
    effective_batch = args.batch_size * grad_accum_steps * world_size
    tokens_per_step = effective_batch * args.context_size
    flops_per_token = 6 * num_params
    flops_per_step = 2 * flops_per_token * tokens_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics (CUMULATIVE mode) ──")
        print(f"  Total parameters:           {num_params:,}")
        print(f"  Trainable parameters:       {num_trainable:,}")
        print(f"  Cumulative keyed params:    {cumulative_swappable['total']:,} "
              f"({swappable_pct_of_max:.2f}% of max swappable)")
        print(f"    - attention:              {cumulative_swappable['attention']:,}")
        print(f"    - mlp:                    {cumulative_swappable['mlp']:,}")
        for i, tier in enumerate(tiers):
            ts = per_tier_swappable[i]
            print(f"    C{tier.tier_id} keyed:             {ts['total']:,}")
        print(f"  Max swappable params:       {max_swappable_params['total']:,}")
        print(f"  Tokens/step:                {tokens_per_step:,}")
        print(f"  FLOPs/step (est):           {flops_per_step:.3e}  (2 passes × 6N × tokens)")
        print(f"  GPU:                        {gpu_name}")
        print(f"  Tier sampling:              {args.tier_sample}")
        print()

    cumulative_tokens = global_step * tokens_per_step
    train_start_wall = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        tier_config = {}
        for i, tier in enumerate(tiers):
            ts = per_tier_swappable[i]
            tier_config[f"compute/swappable_params_c{tier.tier_id}"] = ts["total"]
        wandb.config.update({
            "compute/mode": "cumulative",
            "compute/num_params": num_params,
            "compute/cumulative_keyed_params": cumulative_swappable["total"],
            "compute/cumulative_keyed_attention": cumulative_swappable["attention"],
            "compute/cumulative_keyed_mlp": cumulative_swappable["mlp"],
            "compute/swappable_pct_of_max": swappable_pct_of_max,
            "compute/max_swappable_params": max_swappable_params["total"],
            "compute/tokens_per_step": tokens_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "compute/num_keyed_tiers": num_keyed_tiers,
            **tier_config,
        }, allow_val_change=True)

    # Initial validation
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate_all_tiers_cumulative(
            raw_model, val_dataloader, tiers, device,
            args.eval_steps, is_distributed)
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})

    pbar = tqdm(total=args.max_steps, desc="Cumulative Pretrain",
                initial=global_step) if is_main else None

    # ── Training loop ──
    if local_rank != -1 and global_step > 0:
        sampler.set_epoch(data_epoch)
    data_iter = iter(dataloader)
    # Fast-forward dataloader past batches already consumed in this epoch.
    if args.checkpoint and global_step > 0:
        batches_per_epoch = len(dataloader)
        steps_per_epoch = batches_per_epoch // grad_accum_steps
        if steps_per_epoch > 0:
            batches_consumed = (global_step % steps_per_epoch) * grad_accum_steps
            if batches_consumed > 0 and batches_consumed < batches_per_epoch:
                if is_main:
                    print(f"  Fast-forwarding dataloader: skipping {batches_consumed} batches "
                          f"({batches_consumed}/{batches_per_epoch} in epoch {data_epoch})")
                for _ in range(batches_consumed):
                    next(data_iter)

    while global_step < args.max_steps:
        optimizer.zero_grad()
        model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        # ── Sample which keyed tier to train ──
        active_tier = sample_tier(tiers, args.tier_sample, global_step, tier_rng)
        active_idx = active_tier.tier_idx  # 0-based

        # ── Buffer micro-batches ──
        micro_batches = []
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_epoch += 1
                if local_rank != -1:
                    sampler.set_epoch(data_epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            micro_batches.append(batch)

        total_loss_c1 = 0.0
        total_loss_ck = 0.0
        total_acc_c1 = 0.0
        total_acc_ck = 0.0

        # ==================== PHASE 1: C1 (public, no keys) ====================
        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = (micro_idx == grad_accum_steps - 1)
            sync_ctx = (nullcontext() if (not is_distributed or is_last_micro)
                        else model.no_sync())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_c1 = model(batch["input_ids"], labels=batch["labels"])
                    loss_c1 = outputs_c1.loss
                with torch.no_grad():
                    preds = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
                    targets = batch["labels"][:, 1:]
                    mask = targets != -100
                    acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
                    total_acc_c1 += acc
                    total_loss_c1 += loss_c1.item()
                (loss_c1 * loss_scale).backward()

        # Zero C1 gradients on ALL tiers' keyed weights.
        # These neurons should learn ONLY from their cumulative tier pass.
        for tier in tiers:
            mask_keyed_gradients(raw_model, tier.key, plan=tier.mask_plan)

        # ==================== PHASE 2: C_{k+1} (cumulative keys 0..k) ====================

        # Apply keys 0..active_idx cumulatively
        apply_keys_cumulative(raw_model, tiers, active_idx)

        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = (micro_idx == grad_accum_steps - 1)
            sync_ctx = (nullcontext() if (not is_distributed or is_last_micro)
                        else model.no_sync())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_ck = model(batch["input_ids"], labels=batch["labels"])
                    loss_ck = outputs_ck.loss
                with torch.no_grad():
                    preds = outputs_ck.logits[:, :-1, :].argmax(dim=-1)
                    targets = batch["labels"][:, 1:]
                    mask = targets != -100
                    acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
                    total_acc_ck += acc
                    total_loss_ck += loss_ck.item()
                (loss_ck * loss_scale).backward()

        # ── Gradient combination (CUMULATIVE-AWARE) ──
        #
        # After Phase 1 masking + Phase 2 backward, .grad contains:
        #   Public weights:     grad_c1/A + grad_c_{k+1}/A
        #   Key_k weights:      grad_c_{k+1}/A        (C1 contribution was zeroed)
        #   Key_j weights (j≠k): grad_c_{k+1}/A       (C1 contribution was zeroed)
        #
        # CRITICAL DIFFERENCE from independent mode:
        #   Key_j (j≠k) neurons received C_{k+1} gradients while in a scrambled
        #   configuration (keys 0..k applied). These grads must NOT be applied
        #   at C1 positions — they'd push weights in the wrong direction.
        #
        # Fix: zero ALL keyed grads except the active tier's.
        for tier in tiers:
            if tier.tier_idx != active_idx:
                mask_keyed_gradients(raw_model, tier.key, plan=tier.mask_plan)

        # Average the public portion (same as independent mode)
        scale_public_gradients(raw_model, active_tier.key, scale=0.5,
                               plan=active_tier.mask_plan)

        # ==================== PHASE 3: Return to C1 before optimizer ====================
        #
        # CRITICAL: Adam's per-position momentum (m) and variance (v) must always
        # see weights in C1 arrangement. With rotating cumulative keys, stepping in
        # C_{k+1} arrangement would corrupt optimizer state.
        #
        # 1. Unapply all cumulative keys → weights back to C1 positions
        # 2. Swap gradients → key_k's grads move from C_{k+1} tensor positions
        #    back to C1 tensor positions. Other keyed grads are zero so their
        #    swaps are no-ops.
        unapply_keys_cumulative(raw_model, tiers, active_idx)
        swap_gradients_cumulative(raw_model, tiers, active_idx)

        # ── Clip, step, schedule (all in C1 frame) ──
        # Non-active tiers' keyed positions have zero-tensor grads (zeroed by
        # mask_keyed_gradients after the cumulative backward).  Use
        # adamw_step_preserving_public to prevent weight decay and residual
        # momentum from updating those positions.
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        adamw_step_preserving_public(optimizer, active_tier.adamw_update_masks)
        scheduler.step()

        global_step += 1

        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step

        # ── Logging ──
        avg_loss_c1 = total_loss_c1 / grad_accum_steps
        avg_loss_ck = total_loss_ck / grad_accum_steps
        avg_acc_c1 = total_acc_c1 / grad_accum_steps
        avg_acc_ck = total_acc_ck / grad_accum_steps

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix({
                "c1": f"{avg_loss_c1:.3f}",
                f"c{active_tier.tier_id}": f"{avg_loss_ck:.3f}",
                "tok/s": f"{tps:,.0f}",
            })

        if is_main and global_step % args.log_interval == 0:
            ppl_c1 = math.exp(min(avg_loss_c1, 100))
            ppl_ck = math.exp(min(avg_loss_ck, 100))
            lr = optimizer.param_groups[0]["lr"]
            tier_label = f"c{active_tier.tier_id}"

            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                "loss_c1": avg_loss_c1,
                f"loss_{tier_label}": avg_loss_ck,
                "loss_avg": (avg_loss_c1 + avg_loss_ck) / 2,
                "acc_c1": avg_acc_c1,
                f"acc_{tier_label}": avg_acc_ck,
                "ppl_c1": ppl_c1,
                f"ppl_{tier_label}": ppl_ck,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "sampled_tier": active_tier.tier_id,
                "cumulative_keys_applied": active_idx + 1,
                "train/step": global_step,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/mfu": mfu,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": 2 * flops_per_token * cumulative_tokens,
                "perf/cumulative_petaflops": (2 * flops_per_token * cumulative_tokens) / 1e15,
            }
            log_dict.update(get_gpu_memory_stats(device))
            for tier in tiers:
                log_dict[f"tier_samples/c{tier.tier_id}"] = tier.steps_sampled
            wandb.log(log_dict)

        # ── Validation ──
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            if args.eval_all_tiers:
                val_metrics = evaluate_all_tiers_cumulative(
                    raw_model, val_dataloader, tiers, device,
                    args.eval_steps, is_distributed)
            else:
                eval_batches = _prefetch_eval_batches(
                    val_dataloader, device, args.eval_steps)
                val_metrics = evaluate_tier_cumulative(
                    raw_model, eval_batches, tiers, active_idx, device,
                    is_distributed)
            if is_main:
                wandb.log({**val_metrics, "train/step": global_step})

        # ── Checkpoint ──
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                raw_model, tokenizer, optimizer, save_path,
                scheduler=scheduler, global_step=global_step,
                wandb_run_id=wandb_run_id,
                tier_step_counts=[t.steps_sampled for t in tiers],
                cumulative_wall_secs=cumulative_wall_secs,
                data_epoch=data_epoch)
            print(f"Saved checkpoint to {save_path}")

    if pbar is not None:
        pbar.close()

    # ── Final save ──
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(
            raw_model, tokenizer, optimizer, save_path,
            scheduler=scheduler, global_step=global_step,
            wandb_run_id=wandb_run_id,
            tier_step_counts=[t.steps_sampled for t in tiers],
            cumulative_wall_secs=cumulative_wall_secs,
            data_epoch=data_epoch)

        total_flops = 2 * flops_per_token * cumulative_tokens
        print(f"\n{'='*60}")
        print(f"CUMULATIVE TRAINING COMPLETE — COMPUTE SUMMARY")
        print(f"{'='*60}")
        print(f"  Mode:                  CUMULATIVE")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Keyed tiers:           {num_keyed_tiers}")
        print(f"  Total tokens (D):      {cumulative_tokens:,}")
        print(f"  Total FLOPs:           {total_flops:.4e}")
        print(f"  Total PetaFLOPs:       {total_flops / 1e15:.2f}")
        print(f"  Wall clock (train):    {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):    {(time.time() - train_start_wall) / 3600:.2f} hours")
        print(f"  Avg tokens/sec:        {cumulative_tokens / cumulative_wall_secs:,.0f}")
        if gpu_peak_flops > 0:
            avg_mfu = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops
            print(f"  Avg MFU:               {avg_mfu:.2%}")
        print(f"  GPU:                   {gpu_name} x {world_size}")
        print(f"  Checkpoint:            {save_path}")
        print(f"\n  Tier distribution:")
        for tier in tiers:
            frac = tier.steps_sampled / global_step * 100
            print(f"    C{tier.tier_id}: {tier.steps_sampled:,} steps ({frac:.1f}%)")
        print(f"\n  Cumulative mapping:")
        for i in range(num_keyed_tiers):
            keys_str = " + ".join(f"key_{j+1}" for j in range(i + 1))
            print(f"    C{i+2} = {keys_str}")
        print(f"{'='*60}\n")

        wandb.run.summary.update({
            "final/mode": "cumulative",
            "final/total_steps": global_step,
            "final/total_tokens": cumulative_tokens,
            "final/total_flops": total_flops,
            "final/total_petaflops": total_flops / 1e15,
            "final/wall_clock_hours": cumulative_wall_secs / 3600,
            "final/num_params": num_params,
            "final/gpu_name": gpu_name,
            "final/num_gpus": world_size,
        })
        if gpu_peak_flops > 0:
            wandb.run.summary["final/avg_mfu"] = (
                total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops

        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
