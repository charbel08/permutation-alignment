"""N-Tiered Alignment Pretraining Script.

Extends the 2-tier (C1/C2) joint pretraining to N tiers while keeping
the same compute budget: exactly 2 forward+backward passes per step.

Each step:
1. Forward+backward on C1 (public architecture) — always
2. Sample one keyed tier C_k uniformly from {C2, ..., CN}
3. Forward+backward on C_k
4. Update weights:
   - S'_k (keyed weights for tier k): gradient from C_k only
   - S   (public weights): average of C1 and C_k gradients

Over training, each tier k's private weights see ~1/(N-1) of steps.
Public weights see every step (from C1 + whichever tier was sampled).

Usage:
  torchrun --nproc_per_node=4 pretrain_ntier.py \
    --data_path ./data/tokenized \
    --output_dir ./checkpoints \
    --key_paths key1.json key2.json key3.json \
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
from tiered.train.utils import load_model, save_checkpoint


# ---------------------------------------------------------------------------
# Compute-metrics helpers
# ---------------------------------------------------------------------------

def count_total_parameters(model) -> int:
    """Return the exact total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Return the exact total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_swappable_parameters(model, mask_plan) -> dict:
    """Return the exact number of parameters touched by the current key."""
    from tiered.permutation.utils import _get_attention_module, _get_mlp_module

    total_attn = 0
    total_mlp = 0

    # Attention swappable params
    for layer_idx, idx in mask_plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        n_idx = int(idx.numel())

        # q/k/v selected rows
        total_attn += n_idx * attn.q_proj.weight.shape[1]
        total_attn += n_idx * attn.k_proj.weight.shape[1]
        total_attn += n_idx * attn.v_proj.weight.shape[1]

        # out selected columns
        total_attn += attn.out_proj.weight.shape[0] * n_idx

    # MLP swappable params
    for layer_idx, idx in mask_plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        n_idx = int(idx.numel())

        # c_fc selected rows
        total_mlp += n_idx * mlp.c_fc.weight.shape[1]

        # c_fc selected bias entries
        if mlp.c_fc.bias is not None:
            total_mlp += n_idx

        # c_proj selected columns
        total_mlp += mlp.c_proj.weight.shape[0] * n_idx

    return {
        "total": total_attn + total_mlp,
        "attention": total_attn,
        "mlp": total_mlp,
    }


def count_max_swappable_parameters(model) -> dict:
    """Return the exact number of parameters swappable under a 100% key."""
    from tiered.permutation.utils import _get_attention_module, _get_mlp_module

    total_attn = 0
    total_mlp = 0
    num_layers = len(model.transformer.h)

    for layer_idx in range(num_layers):
        attn = _get_attention_module(model, layer_idx)
        mlp = _get_mlp_module(model, layer_idx)

        # Attention
        total_attn += attn.q_proj.weight.numel()
        total_attn += attn.k_proj.weight.numel()
        total_attn += attn.v_proj.weight.numel()
        total_attn += attn.out_proj.weight.numel()

        # MLP
        total_mlp += mlp.c_fc.weight.numel()
        if mlp.c_fc.bias is not None:
            total_mlp += mlp.c_fc.bias.numel()
        total_mlp += mlp.c_proj.weight.numel()

    return {
        "total": total_attn + total_mlp,
        "attention": total_attn,
        "mlp": total_mlp,
    }


def estimate_flops_per_token(num_layers: int, hidden_size: int, intermediate_size: int,
                              context_size: int, vocab_size: int,
                              num_params: int) -> dict:
    """Estimate FLOPs for a single token in a forward pass (decoder-only transformer).

    Uses the standard breakdown from Kaplan et al. / Hoffmann et al.:
      - Attention QKV projection:   2 * 3 * H^2       per layer
      - Attention output projection: 2 * H^2           per layer
      - Attention logits (Q·K^T):   2 * S * H          per layer  (data-dependent)
      - Attention weighted sum (A·V): 2 * S * H        per layer  (data-dependent)
      - MLP (up + down):            2 * 2 * H * I      per layer
      - Embedding / LM-head:        2 * H * V          (once)

    The factor of 2 in each term accounts for multiply-accumulate = 2 FLOPs.
    Backward pass ≈ 2× forward, so one full fwd+bwd ≈ 3× forward FLOPs.

    Returns a dict with per-component and total FLOPs so you can report the
    breakdown in your paper.
    """
    L, H, I, S, V = num_layers, hidden_size, intermediate_size, context_size, vocab_size

    attn_qkv   = 2 * 3 * H * H         # per layer per token
    attn_out   = 2 * H * H              # per layer per token
    attn_score = 2 * S * H              # per layer per token (avg over positions: S/2, but we use S for peak)
    attn_agg   = 2 * S * H              # per layer per token
    mlp        = 2 * 2 * H * I          # per layer per token (up-proj + down-proj)
    per_layer  = attn_qkv + attn_out + attn_score + attn_agg + mlp
    all_layers = per_layer * L
    embed_lmhead = 2 * H * V            # embedding lookup + LM head projection

    fwd_per_token = all_layers + embed_lmhead

    return {
        "fwd_per_token": fwd_per_token,
        "fwd_bwd_per_token": fwd_per_token * 3,       # fwd + bwd ≈ 3× fwd
        "per_layer_per_token": per_layer,
        "breakdown": {
            "attn_qkv": attn_qkv * L,
            "attn_out": attn_out * L,
            "attn_score": attn_score * L,
            "attn_agg": attn_agg * L,
            "mlp": mlp * L,
            "embed_lmhead": embed_lmhead,
        },
        # Simpler Chinchilla-style approximation for cross-check: 6 * N
        "approx_6N": 6 * num_params,
    }


# Peak bf16 FLOP/s for common GPUs. Used for MFU calculation.
# Values are *tensor-core* peak, not boost-clock peak.
_GPU_PEAK_TFLOPS_BF16 = {
    "A100":  312e12,
    "A10G":   70e12,    # g5 instances
    "H100":  990e12,    # SXM variant
    "H200": 990e12,     # same SM as H100
    "L4":   121e12,
    "L40":  181e12,
    "L40S": 366e12,
    "4090": 330e12,
    "3090": 142e12,
    "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    """Return (peak_flops_bf16, gpu_name) for the current GPU.

    Tries to match the GPU name against known specs. Falls back to 0
    (MFU will be reported as N/A).
    """
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower().replace(" ", ""):
            return peak, name
    return 0.0, name


def get_gpu_memory_stats(device: torch.device) -> dict:
    """Snapshot of current GPU memory usage."""
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
    tier_id: int          # 2, 3, ..., N  (tier 1 is always C1/public)
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan
    # Per-tier tracking
    steps_sampled: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description="N-Tiered Alignment Pretraining")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to tokenized dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")

    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--intermediate_size", type=int, default=None,
                        help="MLP hidden dimension (defaults to 4x hidden_size)")
    parser.add_argument("--untie_weights", action="store_true",
                        help="Disable weight tying between embeddings and LM head")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Permutation keys — one per keyed tier
    parser.add_argument("--key_paths", type=str, nargs="+", required=True,
                        help="Paths to JSON permutation key files (one per keyed tier)")

    # Sampling strategy
    parser.add_argument("--tier_sample", type=str, default="uniform",
                        choices=["uniform", "round_robin"],
                        help="How to pick which keyed tier to train each step")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate before optimizer step")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum LR for cosine schedule (default: 10%% of peak)")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Validation interval in steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Number of batches for validation")
    parser.add_argument("--eval_all_tiers", action="store_true",
                        help="Evaluate ALL tiers at each eval (slower but complete)")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes per rank (reduce on "
                             "machines with few cores per GPU)")

    return parser.parse_args()


def sample_tier(tiers: list[TierInfo], strategy: str, global_step: int,
                rng: random.Random) -> TierInfo:
    """Pick which keyed tier to train this step.
    
    Args:
        tiers: List of TierInfo for keyed tiers (C2..CN).
        strategy: 'uniform' for random, 'round_robin' for deterministic cycling.
        global_step: Current training step (used for round_robin seeding).
        rng: Dedicated Random instance (isolated from library/global state).
    
    Returns:
        The selected TierInfo.
    """
    if strategy == "round_robin":
        tier = tiers[global_step % len(tiers)]
    else:  # uniform
        tier = rng.choice(tiers)
    tier.steps_sampled += 1
    return tier


@torch.inference_mode()
def evaluate_tier(model, eval_batches: list[dict], key, device, is_distributed,
                  swap_plan, tier_label="c2"):
    """Evaluate model for a single tier (C1 + one keyed config).

    Args:
        eval_batches: Pre-fetched list of batch dicts (shared across tiers for
                      fair comparison). Each dict has 'input_ids' and 'labels'
                      already on `device`.
    
    Returns dict with keys prefixed by tier_label.
    """
    model.eval()

    total_loss_c1 = 0.0
    total_loss_ck = 0.0
    total_acc_c1 = 0.0
    total_acc_ck = 0.0
    count = 0

    for batch in eval_batches:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Evaluate C1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        acc_c1 = (preds_c1[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0

        # Evaluate keyed tier
        apply_permutation(model, key, plan=swap_plan)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_ck = model(input_ids, labels=labels)
        loss_ck = outputs_ck.loss.item()
        preds_ck = outputs_ck.logits[:, :-1, :].argmax(dim=-1)
        acc_ck = (preds_ck[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
        unapply_permutation(model, key, plan=swap_plan)

        total_loss_c1 += loss_c1
        total_loss_ck += loss_ck
        total_acc_c1 += acc_c1
        total_acc_ck += acc_ck
        count += 1

    model.train()

    if count == 0:
        return {}

    avg = lambda t: t / count
    vals = [avg(total_loss_c1), avg(total_loss_ck), avg(total_acc_c1), avg(total_acc_ck)]

    if is_distributed:
        metrics_tensor = torch.tensor(vals, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        vals = metrics_tensor.tolist()

    avg_loss_c1, avg_loss_ck, avg_acc_c1, avg_acc_ck = vals
    return {
        "val/loss_c1": avg_loss_c1,
        f"val/loss_{tier_label}": avg_loss_ck,
        "val/acc_c1": avg_acc_c1,
        f"val/acc_{tier_label}": avg_acc_ck,
        "val/ppl_c1": math.exp(min(avg_loss_c1, 100)),
        f"val/ppl_{tier_label}": math.exp(min(avg_loss_ck, 100)),
    }


def _prefetch_eval_batches(dataloader, device: torch.device,
                           num_steps: int) -> list[dict]:
    """Fetch and pin eval batches once so every tier sees identical data."""
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


def evaluate_all_tiers(model, dataloader, tiers, device, num_steps, is_distributed):
    """Evaluate C1 and every keyed tier on *identical* batches.

    Batches are fetched once and reused for each tier so cross-tier loss
    comparisons are not confounded by different eval data.
    """
    eval_batches = _prefetch_eval_batches(dataloader, device, num_steps)
    merged = {}
    for tier in tiers:
        label = f"c{tier.tier_id}"
        metrics = evaluate_tier(
            model, eval_batches, tier.key, device,
            is_distributed, tier.swap_plan, tier_label=label
        )
        merged.update(metrics)
    return merged


def train(args):
    """Main training loop."""
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

    # Dedicated RNG for tier sampling — isolated from global/library state
    # so the schedule is reproducible regardless of other random calls.
    tier_rng = random.Random(42)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load keys and build per-tier plans ──
    num_keyed_tiers = len(args.key_paths)
    if is_main:
        print(f"Setting up {num_keyed_tiers} keyed tier(s) + C1 (public) "
              f"= {num_keyed_tiers + 1} total tiers")

    # ── Load model ──
    if args.checkpoint:
        model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    else:
        model = load_model(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            context_size=args.context_size,
            intermediate_size=args.intermediate_size,
            tie_weights=not args.untie_weights,
            do_print=is_main,
        )

    model.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Build plans (needs raw model before compile)
    tiers: list[TierInfo] = []
    for i, key_path in enumerate(args.key_paths):
        key = load_key(key_path)
        swap_plan = build_swap_plan(model, key, device)
        mask_plan = build_mask_plan(model, key, device)
        tier_id = i + 2  # tier 1 is C1 (public)
        tiers.append(TierInfo(
            tier_id=tier_id, key=key,
            swap_plan=swap_plan, mask_plan=mask_plan,
        ))
        if is_main:
            print(f"  Tier C{tier_id}: key={key_path}, "
                  f"{len(key.attn_heads)} attn swaps, {len(key.mlp_cols)} MLP swaps, "
                  f"{len(swap_plan.attn_ops)} swap ops")

    raw_model = model
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

    if local_rank != -1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        collate_fn=collator, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            sampler=val_sampler, shuffle=False,
            collate_fn=collator, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ── Optimizer & scheduler ──
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), fused=True)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )

    # ── Resume ──
    global_step = 0
    wandb_run_id = None
    tier_step_counts = None  # restored from checkpoint if available
    if args.checkpoint:
        training_state_path = os.path.join(args.checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"Checkpoint dir exists but training_state.pt not found: {training_state_path}"
            )
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = training_state["global_step"]
        wandb_run_id = training_state.get("wandb_run_id")
        tier_step_counts = training_state.get("tier_step_counts")

        # Restore per-tier sample counts so logging is continuous
        if tier_step_counts and len(tier_step_counts) == len(tiers):
            for tier, count in zip(tiers, tier_step_counts):
                tier.steps_sampled = count

        # Re-seed the dedicated RNG to the same sequence position on resume.
        # Advance it by global_step draws so we get the same tier schedule
        # as if we'd trained from scratch (for uniform sampling).
        tier_rng = random.Random(42)
        for _ in range(global_step):
            tier_rng.randint(0, num_keyed_tiers - 1)

        if is_main:
            print(f"Resumed training state from step {global_step}")

    # ── Wandb ──
    if is_main:
        if args.checkpoint and wandb_run_id:
            wandb.init(project=args.wandb_project, id=wandb_run_id,
                       resume="must", config=vars(args))
        else:
            wandb.init(project=args.wandb_project, name=args.run_name,
                       config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # ── Training loop ──
    # Monotonic epoch counter for DistributedSampler shuffling. Using a
    # dedicated counter (rather than global_step) gives deterministic
    # shuffle order regardless of batch size or dataset size changes.
    data_epoch = 0
    if local_rank != -1 and global_step > 0:
        sampler.set_epoch(data_epoch)
    data_iter = iter(dataloader)

    is_distributed = (local_rank != -1)
    grad_accum_steps = args.grad_accum_steps
    loss_scale = 1.0 / grad_accum_steps
    effective_batch = args.batch_size * grad_accum_steps * world_size

    if is_main:
        print(f"Effective batch size: {args.batch_size} x {grad_accum_steps} "
              f"x {world_size} = {effective_batch}")
        print(f"Tier sampling: {args.tier_sample}, "
              f"each keyed tier sees ~1/{num_keyed_tiers} of steps")

    # ── Compute metrics setup ──
    num_params = count_total_parameters(raw_model)
    num_trainable = count_trainable_parameters(raw_model)
    swappable_params = count_swappable_parameters(raw_model, tiers[0].mask_plan)
    max_swappable_params = count_max_swappable_parameters(raw_model)

    swappable_pct_of_max = 100.0 * swappable_params["total"] / max_swappable_params["total"]
    max_swappable_pct_of_total = 100.0 * max_swappable_params["total"] / num_params

    inter_size = args.intermediate_size or (args.hidden_size * 4)

    # Get the actual vocab size from the model's embedding layer
    vocab_size = raw_model.get_input_embeddings().weight.shape[0]

    flop_info = estimate_flops_per_token(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=inter_size,
        context_size=args.context_size,
        vocab_size=vocab_size,
        num_params=num_params,
    )
    # Tokens processed per optimizer step (across all ranks):
    # batch_size * context_size * grad_accum_steps * world_size
    tokens_per_step = effective_batch * args.context_size
    # Standard C = 6ND (Kaplan et al., Hoffmann et al., used by LLaMA 3, PaLM, etc.)
    # 6N = 2N (fwd, multiply-accumulate) + 4N (bwd, 2× fwd) per token per pass
    flops_per_token = 6 * num_params
    # We do 2 fwd+bwd per step (C1 + sampled C_k)
    flops_per_step = 2 * flops_per_token * tokens_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Total parameters:           {num_params:,}")
        print(f"  Trainable parameters:       {num_trainable:,}")
        print(f"  Current swappable params:   {swappable_params['total']:,} ({swappable_pct_of_max:.2f}% of max swappable)")
        print(f"    - attention:              {swappable_params['attention']:,}")
        print(f"    - mlp:                    {swappable_params['mlp']:,}")
        print(f"  Max swappable params:       {max_swappable_params['total']:,} ({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"    - attention:              {max_swappable_params['attention']:,}")
        print(f"    - mlp:                    {max_swappable_params['mlp']:,}")
        print(f"  Tokens/step:                {tokens_per_step:,}")
        print(f"  FLOPs/token (6N):           {flops_per_token:.3e}")
        print(f"  FLOPs/step (est):           {flops_per_step:.3e}  (2 passes × 6N × tokens)")
        print(f"  GPU:                        {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:              {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:              unknown (MFU will be N/A)")
        # Detailed breakdown for supplementary material
        print(f"  Detailed fwd/token:         {flop_info['fwd_per_token']:.3e}  "
              f"(vs 2N={2*num_params:.3e}, ratio={flop_info['fwd_per_token']/(2*num_params):.3f})")
        print()

    # Cumulative trackers (restored on resume)
    cumulative_tokens = global_step * tokens_per_step
    train_start_wall = time.time()
    cumulative_wall_secs = 0.0  # wall time spent in training steps only
    if args.checkpoint:
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        if is_main and cumulative_wall_secs > 0:
            print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    # Reset peak memory stats so we track from this point
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log static compute info to wandb config (useful for paper tables)
    if is_main:
        wandb.config.update({
            "compute/num_params": num_params,
            "compute/num_trainable_params": num_trainable,
            "compute/swappable_params": swappable_params["total"],
            "compute/swappable_attention_params": swappable_params["attention"],
            "compute/swappable_mlp_params": swappable_params["mlp"],
            "compute/swappable_pct_of_max": swappable_pct_of_max,
            "compute/max_swappable_params": max_swappable_params["total"],
            "compute/max_swappable_attention_params": max_swappable_params["attention"],
            "compute/max_swappable_mlp_params": max_swappable_params["mlp"],
            "compute/max_swappable_pct_of_total": max_swappable_pct_of_total,
            "compute/tokens_per_step": tokens_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/flops_per_token_6N": flops_per_token,
            "compute/flops_per_token_detailed": flop_info["fwd_bwd_per_token"],
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "compute/num_keyed_tiers": num_keyed_tiers,
            "compute/vocab_size": vocab_size,
        }, allow_val_change=True)


    # Initial validation
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate_all_tiers(
            raw_model, val_dataloader, tiers, device,
            args.eval_steps, is_distributed
        )
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})

    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None

    while global_step < args.max_steps:
        optimizer.zero_grad()
        model.train()
        
        # ── Step timing (CUDA-synced for accuracy) ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        # ── Sample which keyed tier to train this step ──
        active_tier = sample_tier(tiers, args.tier_sample, global_step, tier_rng)

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

        # ==================== PHASE 1: C1 (public) ====================
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

        # Zero C1 gradients on this tier's keyed weights.
        # NOTE: This operates on raw_model (unwrapped), which is safe because
        # DDP's allreduce hooks fire during the final synced backward above.
        # By this point .grad tensors are already reduced across ranks, so
        # subsequent in-place modifications (masking, scaling, swapping) are
        # purely local and don't interact with DDP's communication.
        mask_keyed_gradients(raw_model, active_tier.key, plan=active_tier.mask_plan)

        # ==================== PHASE 2: C_k (sampled keyed tier) ====================
        apply_permutation(raw_model, active_tier.key, plan=active_tier.swap_plan)

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

        # ── Gradient combination ──
        # Each backward pass scales loss by loss_scale = 1/grad_accum_steps, so
        # after accumulation each phase contributes grad/grad_accum_steps.
        #
        # After Phase 1 masking + Phase 2 backward, the accumulated .grad is:
        #   Public weights:  grad_c1/A + grad_ck/A   (A = grad_accum_steps)
        #   Keyed weights:   grad_ck/A               (C1 contribution was zeroed)
        #
        # scale_public_gradients(..., 0.5) then halves only the public portion:
        #   Public weights:  (grad_c1 + grad_ck) / (2A)   ← averaged over both phases
        #   Keyed weights:   grad_ck / A                   ← unchanged, full contribution
        #
        # This means public weights see the mean of both configurations' gradients,
        # while keyed weights get the unattenuated gradient from their own config.
        scale_public_gradients(raw_model, active_tier.key, scale=0.5,
                               plan=active_tier.mask_plan)

        # ==================== PHASE 3: return to C1 before optimizer ====================
        # CRITICAL for N-tier: Adam's per-position momentum (m) and variance (v)
        # must always see weights in C1 arrangement. With rotating keys, stepping
        # in C_k arrangement would apply key_a's momentum to key_b's positions
        # on the next step. unapply moves weights back to C1, then swap_gradients
        # moves the C_k-phase gradients into the matching C1 positions.
        unapply_permutation(raw_model, active_tier.key, plan=active_tier.swap_plan)
        swap_gradients(raw_model, active_tier.key, plan=active_tier.swap_plan)

        # ── Clip, step, schedule (all in C1 frame) ──
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        optimizer.step()
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

            # ── Throughput metrics ──
            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            samples_per_sec = effective_batch / step_elapsed if step_elapsed > 0 else 0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0
            # MFU: fraction of hardware peak actually used (per-GPU)
            # achieved_flops_per_sec is total across all GPUs; divide by world_size
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                # ── Task losses ──
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
                "train/step": global_step,
                # ── Timing ──
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": (time.time() - train_start_wall) / 3600,
                # ── Throughput ──
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/tokens_per_sec_per_gpu": tokens_per_sec / world_size,
                "perf/samples_per_sec": samples_per_sec,
                # ── FLOPs ──
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/achieved_tflops_per_gpu": per_gpu_flops / 1e12,
                "perf/mfu": mfu,
                # ── Cumulative (for paper: total compute budget, C = 6ND) ──
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": 2 * flops_per_token * cumulative_tokens,
                "perf/cumulative_petaflops": (2 * flops_per_token * cumulative_tokens) / 1e15,
            }
            # GPU memory
            log_dict.update(get_gpu_memory_stats(device))
            # Tier sampling distribution
            for tier in tiers:
                log_dict[f"tier_samples/c{tier.tier_id}"] = tier.steps_sampled
            wandb.log(log_dict)

        # ── Validation ──
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            # Prefetch eval batches once; reused for all tiers if eval_all_tiers
            eval_batches = _prefetch_eval_batches(
                val_dataloader, device, args.eval_steps
            )
            if args.eval_all_tiers:
                # Full eval: every tier on identical batches
                val_metrics = evaluate_all_tiers(
                    raw_model, val_dataloader, tiers, device,
                    args.eval_steps, is_distributed
                )
            else:
                # Cheap eval: only the tier we just trained
                val_metrics = evaluate_tier(
                    raw_model, eval_batches, active_tier.key, device,
                    is_distributed, active_tier.swap_plan,
                    tier_label=f"c{active_tier.tier_id}",
                )
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
            )
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
        )
        total_flops = 2 * flops_per_token * cumulative_tokens
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — COMPUTE SUMMARY (for paper)")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Current swappable:     {swappable_params['total']:,} ({swappable_pct_of_max:.2f}% of max swappable)")
        print(f"    - attention:         {swappable_params['attention']:,}")
        print(f"    - mlp:               {swappable_params['mlp']:,}")
        print(f"  Max swappable:         {max_swappable_params['total']:,} ({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"    - attention:         {max_swappable_params['attention']:,}")
        print(f"    - mlp:               {max_swappable_params['mlp']:,}")
        print(f"  Total tokens (D):      {cumulative_tokens:,}")
        print(f"  Total FLOPs (2×6ND):   {total_flops:.4e}")
        print(f"  Total PetaFLOPs:       {total_flops / 1e15:.2f}")
        print(f"  Wall clock (train):    {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):    {(time.time() - train_start_wall) / 3600:.2f} hours")
        print(f"  Avg tokens/sec:        {cumulative_tokens / cumulative_wall_secs:,.0f}")
        print(f"  Avg tokens/sec/GPU:    {cumulative_tokens / cumulative_wall_secs / world_size:,.0f}")
        if gpu_peak_flops > 0:
            avg_mfu = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops
            print(f"  Avg MFU:               {avg_mfu:.2%}")
        print(f"  GPU:                   {gpu_name} x {world_size}")
        print(f"  Checkpoint:            {save_path}")
        print(f"\n  Tier distribution:")
        for tier in tiers:
            frac = tier.steps_sampled / global_step * 100
            print(f"    C{tier.tier_id}: {tier.steps_sampled:,} steps "
                  f"({frac:.1f}%)")
        print(f"{'='*60}\n")

        # Log final summary to wandb for easy access
        wandb.run.summary.update({
            "final/total_steps": global_step,
            "final/total_tokens": cumulative_tokens,
            "final/total_flops": total_flops,
            "final/total_petaflops": total_flops / 1e15,
            "final/wall_clock_hours": cumulative_wall_secs / 3600,
            "final/avg_tokens_per_sec": cumulative_tokens / cumulative_wall_secs,
            "final/avg_tokens_per_sec_per_gpu": cumulative_tokens / cumulative_wall_secs / world_size,
            "final/num_params": num_params,
            "final/swappable_params": swappable_params["total"],
            "final/swappable_attention_params": swappable_params["attention"],
            "final/swappable_mlp_params": swappable_params["mlp"],
            "final/swappable_pct_of_max": swappable_pct_of_max,
            "final/max_swappable_params": max_swappable_params["total"],
            "final/max_swappable_attention_params": max_swappable_params["attention"],
            "final/max_swappable_mlp_params": max_swappable_params["mlp"],
            "final/max_swappable_pct_of_total": max_swappable_pct_of_total,
            "final/gpu_name": gpu_name,
            "final/num_gpus": world_size,
        })
        if gpu_peak_flops > 0:
            wandb.run.summary["final/avg_mfu"] = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)