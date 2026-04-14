"""Tiered Alignment Pretraining Script with periodic C2 passes.

This script implements the joint pretraining of public (C1) and keyed (C2)
architectures with asymmetric gradient updates as described in the paper.

Training loop:
1. Forward pass on C1, get loss l1
2. Backward pass on C1, store gradients for S (public weights)
3. Apply permutation to get C2
4. Forward pass on C2, get loss l2
5. Backward pass on C2, get gradients for both S and S'
6. Update weights:
   - S' <- S' + lr * grad2_S' (keyed weights: only from C2)
   - S  <- S  + lr * 0.5 * (grad1_S + grad2_S) (public weights: average)
7. Unapply permutation to return to C1

Periodic-C2 mode:
- C1 pass always runs.
- C2 pass runs every K optimizer steps (1-indexed): 1, 1+K, 1+2K, ...
- On non-C2 steps, keyed pass and public-gradient averaging are skipped.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, scale_public_gradients
from tiered.permutation.masking import build_mask_plan, mask_keyed_gradients
from tiered.permutation.permute import (
    apply_permutation,
    build_swap_plan,
    swap_gradients,
    unapply_permutation,
)
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
    """Return the exact total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Return the exact total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_swappable_parameters(model, mask_plan) -> dict:
    """Return the exact number of parameters touched by the current key.

    "Swappable" here means parameters that participate in keyed swaps / masking:
      - Attention:
          q_proj/k_proj/v_proj rows for keyed full-attention head dims
          out_proj.weight columns for keyed head dims from either full-attn or
          out-projection-only swaps
      - MLP:
          c_fc.weight rows + c_fc.bias entries for keyed columns from either
          mlp_cols (both) or mlp_up_cols (up-only)
          c_proj.weight columns for keyed columns from either mlp_cols (both)
          or mlp_down_cols (down-only)

    Counts are taken directly from the instantiated tensors, not estimated from
    hyperparameters.
    """
    from tiered.permutation.utils import _get_attention_module, _get_mlp_module

    total_attn = 0
    total_mlp = 0

    def _merge_unique_idx(*idx_tensors):
        parts = [x for x in idx_tensors if x is not None and x.numel() > 0]
        if not parts:
            return None
        return torch.unique(torch.cat(parts, dim=0), sorted=False)

    # Attention swappable params
    attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in attn_layers:
        attn = _get_attention_module(model, layer_idx)
        idx_full = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_out = _merge_unique_idx(idx_full, mask_plan.keyed_attn_out_indices.get(layer_idx))

        # q/k/v selected rows only come from full-attn head swaps.
        if idx_full is not None and idx_full.numel() > 0:
            n_idx_full = int(idx_full.numel())
            total_attn += n_idx_full * attn.q_proj.weight.shape[1]
            total_attn += n_idx_full * attn.k_proj.weight.shape[1]
            total_attn += n_idx_full * attn.v_proj.weight.shape[1]

        # out-proj selected columns come from full-attn or out-only swaps.
        if idx_out is not None and idx_out.numel() > 0:
            n_idx_out = int(idx_out.numel())
            total_attn += attn.out_proj.weight.shape[0] * n_idx_out

    # MLP swappable params
    mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in mlp_layers:
        mlp = _get_mlp_module(model, layer_idx)

        idx_rows = _merge_unique_idx(
            mask_plan.keyed_mlp_indices.get(layer_idx),
            mask_plan.keyed_mlp_up_indices.get(layer_idx),
        )
        idx_cols = _merge_unique_idx(
            mask_plan.keyed_mlp_indices.get(layer_idx),
            mask_plan.keyed_mlp_down_indices.get(layer_idx),
        )

        # c_fc selected rows + bias entries (both + up-only)
        if idx_rows is not None and idx_rows.numel() > 0:
            n_rows = int(idx_rows.numel())
            total_mlp += n_rows * mlp.c_fc.weight.shape[1]
            if mlp.c_fc.bias is not None:
                total_mlp += n_rows

        # c_proj selected columns (both + down-only)
        if idx_cols is not None and idx_cols.numel() > 0:
            n_cols = int(idx_cols.numel())
            total_mlp += mlp.c_proj.weight.shape[0] * n_cols

    return {
        "total": total_attn + total_mlp,
        "attention": total_attn,
        "mlp": total_mlp,
    }


def count_max_swappable_parameters(model) -> dict:
    """Return the exact number of parameters swappable under a 100% key.

    This counts the full permutation-supporting subset of the architecture:
      - Attention:
          q_proj.weight (all rows)
          k_proj.weight (all rows)
          v_proj.weight (all rows)
          out_proj.weight (all columns)
      - MLP:
          c_fc.weight (all rows)
          c_fc.bias (all entries)
          c_proj.weight (all columns)

    This is the maximum number of parameters that could be affected if the key
    covered 100% of swappable units.
    """
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


def estimate_flops_per_token(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    context_size: int,
    vocab_size: int,
    num_params: int,
) -> dict:
    """Estimate FLOPs for a single token in a forward pass (decoder-only transformer).

    Uses the standard breakdown from Kaplan et al. / Hoffmann et al.:
      - Attention QKV projection:    2 * 3 * H^2       per layer
      - Attention output projection: 2 * H^2           per layer
      - Attention logits (Q·K^T):    2 * S * H         per layer
      - Attention weighted sum (A·V): 2 * S * H        per layer
      - MLP (up + down):             2 * 2 * H * I     per layer
      - Embedding / LM-head:         2 * H * V         once

    The factor of 2 in each term accounts for multiply-accumulate = 2 FLOPs.
    Backward pass ≈ 2× forward, so one full fwd+bwd ≈ 3× forward FLOPs.
    """
    L, H, I, S, V = num_layers, hidden_size, intermediate_size, context_size, vocab_size

    attn_qkv = 2 * 3 * H * H
    attn_out = 2 * H * H
    attn_score = 2 * S * H
    attn_agg = 2 * S * H
    mlp = 2 * 2 * H * I
    per_layer = attn_qkv + attn_out + attn_score + attn_agg + mlp
    all_layers = per_layer * L
    embed_lmhead = 2 * H * V

    fwd_per_token = all_layers + embed_lmhead

    return {
        "fwd_per_token": fwd_per_token,
        "fwd_bwd_per_token": fwd_per_token * 3,
        "per_layer_per_token": per_layer,
        "breakdown": {
            "attn_qkv": attn_qkv * L,
            "attn_out": attn_out * L,
            "attn_score": attn_score * L,
            "attn_agg": attn_agg * L,
            "mlp": mlp * L,
            "embed_lmhead": embed_lmhead,
        },
        "approx_6N": 6 * num_params,
    }


# Peak bf16 FLOP/s for common GPUs. Used for MFU calculation.
# Values are tensor-core peak, not boost-clock peak.
_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12,
    "A10G": 70e12,
    "H100": 990e12,
    "H200": 990e12,
    "L4": 121e12,
    "L40": 181e12,
    "L40S": 366e12,
    "4090": 330e12,
    "3090": 142e12,
    "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    """Return (peak_flops_bf16, gpu_name) for the current GPU."""
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    normalized = name.lower().replace(" ", "")
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in normalized:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Tiered Alignment Pretraining")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")

    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=None,
        help="MLP hidden dimension (defaults to 4x hidden_size)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Permutation key
    parser.add_argument("--key_path", type=str, required=True, help="Path to JSON permutation key file")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer step",
    )
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=6e-5,
        help="Minimum LR for cosine schedule (default: 10%% of peak)",
    )
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--c2_every_k",
        type=int,
        default=1,
        help="Run C2 pass every K optimizer steps (1-indexed cadence).",
    )
    parser.add_argument(
        "--c2_metrics_every_k",
        type=int,
        default=None,
        help="On non-C2 steps, run a no-grad C2 forward every K steps purely "
             "to log val-free train metrics (loss_c2/acc_c2). Default: disabled "
             "— C2 metrics are only emitted on steps where the C2 backward runs.",
    )

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500, help="Validation interval in steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of batches for validation")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes per rank (reduce on "
                             "machines with few cores per GPU)")

    args = parser.parse_args()
    if args.c2_every_k < 1:
        parser.error("--c2_every_k must be >= 1")
    if args.c2_metrics_every_k is not None and args.c2_metrics_every_k < 1:
        parser.error("--c2_metrics_every_k must be >= 1 or unset")
    return args


@torch.inference_mode()
def evaluate(model, dataloader, key, device, num_steps=50, is_distributed=False, swap_plan=None):
    """Evaluate model on a dataset, computing C1 and C2 metrics."""
    model.eval()

    total_loss_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c1 = 0.0
    total_acc_c2 = 0.0
    count = 0

    data_iter = iter(dataloader)

    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Evaluate C1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets_c1 = labels[:, 1:]
        mask_c1 = targets_c1 != -100
        acc_c1 = (preds_c1[mask_c1] == targets_c1[mask_c1]).float().mean().item() if mask_c1.any() else 0.0

        # Evaluate C2
        apply_permutation(model, key, plan=swap_plan)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c2 = model(input_ids, labels=labels)
        loss_c2 = outputs_c2.loss.item()
        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets_c2 = labels[:, 1:]
        mask_c2 = targets_c2 != -100
        acc_c2 = (preds_c2[mask_c2] == targets_c2[mask_c2]).float().mean().item() if mask_c2.any() else 0.0
        unapply_permutation(model, key, plan=swap_plan)

        total_loss_c1 += loss_c1
        total_loss_c2 += loss_c2
        total_acc_c1 += acc_c1
        total_acc_c2 += acc_c2
        count += 1

    model.train()

    if count == 0:
        return {
            "val/loss_c1": 0,
            "val/loss_c2": 0,
            "val/acc_c1": 0,
            "val/acc_c2": 0,
            "val/ppl_c1": 0,
            "val/ppl_c2": 0,
        }

    avg_loss_c1 = total_loss_c1 / count
    avg_loss_c2 = total_loss_c2 / count
    avg_acc_c1 = total_acc_c1 / count
    avg_acc_c2 = total_acc_c2 / count

    # All-reduce across ranks to get global average
    if is_distributed:
        metrics_tensor = torch.tensor([avg_loss_c1, avg_loss_c2, avg_acc_c1, avg_acc_c2], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss_c1, avg_loss_c2, avg_acc_c1, avg_acc_c2 = metrics_tensor.tolist()

    return {
        "val/loss_c1": avg_loss_c1,
        "val/loss_c2": avg_loss_c2,
        "val/acc_c1": avg_acc_c1,
        "val/acc_c2": avg_acc_c2,
        "val/ppl_c1": math.exp(min(avg_loss_c1, 100)),
        "val/ppl_c2": math.exp(min(avg_loss_c2, 100)),
    }


def train(args):
    """Main training loop."""
    # Setup distributed
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

    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load key
    key = load_key(args.key_path)
    if is_main:
        print(f"Loaded key with {len(key.attn_heads)} attention swaps, {len(key.mlp_cols)} MLP swaps")

    # Load model
    if args.checkpoint:
        model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    else:
        model = load_model(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            context_size=args.context_size,
            intermediate_size=args.intermediate_size,
            tie_weights=True,
            do_print=is_main,
        )

    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Pre-build swap and mask plans BEFORE torch.compile
    swap_plan = build_swap_plan(model, key, device)
    if is_main:
        print(
            "Built swap plan: "
            f"{len(swap_plan.attn_ops)} full-attn ops, "
            f"{len(swap_plan.attn_out_ops)} out-only-attn ops, "
            f"{len(swap_plan.mlp_ops)} mlp-both ops, "
            f"{len(swap_plan.mlp_up_ops)} mlp-up-only ops, "
            f"{len(swap_plan.mlp_down_ops)} mlp-down-only ops "
            f"(indices on {device})"
        )

    mask_plan = build_mask_plan(model, key, device)
    if is_main:
        n_attn_layers = len(mask_plan.keyed_attn_indices)
        n_attn_out_layers = len(mask_plan.keyed_attn_out_indices)
        n_mlp_layers = len(mask_plan.keyed_mlp_indices)
        n_mlp_up_layers = len(mask_plan.keyed_mlp_up_indices)
        n_mlp_down_layers = len(mask_plan.keyed_mlp_down_indices)
        print(
            "Built mask plan: "
            f"{n_attn_layers} attn layers, {n_attn_out_layers} out-only attn layers, "
            f"{n_mlp_layers} mlp-both layers, {n_mlp_up_layers} mlp-up-only layers, "
            f"{n_mlp_down_layers} mlp-down-only layers with keyed indices"
        )

    # Keep a reference to the original model for permutation/masking ops.
    raw_model = model

    # Precompute update masks for non-C2 steps.
    # On steps where C2 doesn't run, keyed positions receive no gradient.
    # mask_keyed_gradients leaves them as zero tensors (not None), so AdamW
    # would still apply weight decay and residual momentum to them.
    # build_adamw_update_masks([mask_plan]) marks keyed positions as frozen
    # (False); pass to adamw_step_preserving_public instead of optimizer.step().
    c1_only_update_masks = build_adamw_update_masks(raw_model, [mask_plan])

    # Compile the forward pass for fused kernels.
    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])

    # Load data
    full_dataset = load_from_disk(args.data_path)

    if "train" in full_dataset:
        train_dataset = full_dataset["train"]
    else:
        train_dataset = full_dataset

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
        if is_main:
            print(f"Removed columns: {cols_to_remove}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if local_rank != -1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        if local_rank != -1:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Setup optimizer: weight decay on 2D+ params only
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    # LR schedule: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps - args.warmup_steps,
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    # Resume from checkpoint if provided
    global_step = 0
    wandb_run_id = None
    c2_passes_cumulative = 0
    data_epoch = 0
    if args.checkpoint:
        training_state_path = os.path.join(args.checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"Checkpoint dir exists but training_state.pt not found: {training_state_path}\n"
                f"Cannot resume training without optimizer/scheduler state."
            )
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = training_state["global_step"]
        wandb_run_id = training_state.get("wandb_run_id")
        inferred_c2_passes = ((global_step - 1) // args.c2_every_k + 1) if global_step > 0 else 0
        c2_passes_cumulative = int(training_state.get("c2_passes_cumulative", inferred_c2_passes))
        data_epoch = training_state.get("data_epoch", 0)
        if is_main:
            print(f"Resumed training state from step {global_step}")
            print(f"  Resumed data_epoch: {data_epoch}")

    # Setup wandb
    if is_main:
        if args.checkpoint and wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume="allow",
                config=vars(args),
            )
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # Training loop setup
    # Monotonic epoch counter for DistributedSampler shuffling. Using a
    # dedicated counter (rather than global_step) gives deterministic
    # shuffle order regardless of batch size or dataset size changes.
    if local_rank != -1 and global_step > 0:
        sampler.set_epoch(data_epoch)
    data_iter = iter(dataloader)

    is_distributed = local_rank != -1
    grad_accum_steps = args.grad_accum_steps
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
    loss_scale = 1.0 / grad_accum_steps
    effective_batch = args.batch_size * grad_accum_steps * world_size

    if is_main:
        print(f"Effective batch size: {args.batch_size} x {grad_accum_steps} x {world_size} = {effective_batch}")

    # ── Compute metrics setup ──
    num_params = count_total_parameters(raw_model)
    num_trainable = count_trainable_parameters(raw_model)
    swappable_params = count_swappable_parameters(raw_model, mask_plan)
    max_swappable_params = count_max_swappable_parameters(raw_model)

    swappable_pct_of_max = 100.0 * swappable_params["total"] / max_swappable_params["total"]
    swappable_pct_of_total = 100.0 * swappable_params["total"] / num_params
    max_swappable_pct_of_total = 100.0 * max_swappable_params["total"] / num_params

    inter_size = args.intermediate_size or (args.hidden_size * 4)
    vocab_size = raw_model.get_input_embeddings().weight.shape[0]

    flop_info = estimate_flops_per_token(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=inter_size,
        context_size=args.context_size,
        vocab_size=vocab_size,
        num_params=num_params,
    )

    # Tokens processed per optimizer step (across all ranks)
    tokens_per_step = effective_batch * args.context_size

    # Standard C = 6ND per fwd+bwd pass.
    flops_per_token = 6 * num_params
    flops_step_baseline = flops_per_token * tokens_per_step
    flops_step_with_c2 = 2 * flops_per_token * tokens_per_step
    expected_flops_per_step = flops_step_baseline * (1.0 + 1.0 / args.c2_every_k)

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Total parameters:           {num_params:,}")
        print(f"  Trainable parameters:       {num_trainable:,}")
        print(
            "  Current swapped params:     "
            f"{swappable_params['total']:,} "
            f"({swappable_pct_of_max:.2f}% of swappable, {swappable_pct_of_total:.2f}% of total params)"
        )
        print(f"    - attention:              {swappable_params['attention']:,}")
        print(f"    - mlp:                    {swappable_params['mlp']:,}")
        print(f"  Max swappable params:       {max_swappable_params['total']:,} ({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"    - attention:              {max_swappable_params['attention']:,}")
        print(f"    - mlp:                    {max_swappable_params['mlp']:,}")
        print(f"  Tokens/step:                {tokens_per_step:,}")
        print(f"  FLOPs/token (6N):           {flops_per_token:.3e}")
        print(f"  FLOPs/step (C1 only):       {flops_step_baseline:.3e}")
        print(f"  FLOPs/step (C1 + C2):       {flops_step_with_c2:.3e}")
        print(f"  Expected FLOPs/step:        {expected_flops_per_step:.3e}  (C2 every K={args.c2_every_k})")
        print(f"  GPU:                        {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:              {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:              unknown (MFU will be N/A)")
        print(
            f"  Detailed fwd/token:         {flop_info['fwd_per_token']:.3e}  "
            f"(vs 2N={2 * num_params:.3e}, ratio={flop_info['fwd_per_token'] / (2 * num_params):.3f})"
        )
        print()

    # Cumulative trackers
    cumulative_tokens = global_step * tokens_per_step
    cumulative_flops = float((global_step + c2_passes_cumulative) * flops_step_baseline)
    train_start_wall = time.time()
    cumulative_wall_secs = 0.0
    if args.checkpoint:
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        cumulative_flops = float(training_state.get("cumulative_flops", cumulative_flops))
        if is_main and cumulative_wall_secs > 0:
            print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")
    # Reset peak memory stats so we track from this point
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log static compute info to wandb config
    if is_main:
        wandb.config.update(
            {
                "compute/total_params": num_params,
                "compute/trainable_params": num_trainable,
                "compute/swappable_params": swappable_params["total"],
                "compute/swappable_attention_params": swappable_params["attention"],
                "compute/swappable_mlp_params": swappable_params["mlp"],
                "compute/swappable_pct_of_max": swappable_pct_of_max,
                "compute/swappable_pct_of_total": swappable_pct_of_total,
                "compute/max_swappable_params": max_swappable_params["total"],
                "compute/max_swappable_attention_params": max_swappable_params["attention"],
                "compute/max_swappable_mlp_params": max_swappable_params["mlp"],
                "compute/max_swappable_pct_of_total": max_swappable_pct_of_total,
                "compute/tokens_per_step": tokens_per_step,
                "compute/c2_every_k": args.c2_every_k,
                "compute/flops_per_step": expected_flops_per_step,
                "compute/flops_per_step_baseline": flops_step_baseline,
                "compute/flops_per_step_with_c2": flops_step_with_c2,
                "compute/flops_per_token_6N": flops_per_token,
                "compute/flops_per_token_detailed": flop_info["fwd_bwd_per_token"],
                "compute/gpu_name": gpu_name,
                "compute/gpu_peak_bf16_flops": gpu_peak_flops,
                "compute/vocab_size": vocab_size,
            },
            allow_val_change=True,
        )

    # Initial validation at step 0
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate(
            raw_model,
            val_dataloader,
            key,
            device,
            args.eval_steps,
            is_distributed=is_distributed,
            swap_plan=swap_plan,
        )
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})

    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None

    while global_step < args.max_steps:
        optimizer.zero_grad()

        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        total_loss_c1 = 0.0
        total_loss_c2 = 0.0
        total_acc_c1 = 0.0
        total_acc_c2 = 0.0
        step_num = global_step + 1
        do_c2 = ((step_num - 1) % args.c2_every_k) == 0
        want_c2_metrics = (
            not do_c2
            and args.c2_metrics_every_k is not None
            and ((step_num - 1) % args.c2_metrics_every_k) == 0
        )
        # True iff loss_c2 / acc_c2 were computed this step (from backward or
        # the no-grad metrics pass). Drives conditional logging below.
        have_c2_metrics = do_c2 or want_c2_metrics

        model.train()

        # Buffer micro-batches so C1 and C2 use identical data
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

        # ==================== STEP 1: PUBLIC ARCHITECTURE (C1) ====================
        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = micro_idx == grad_accum_steps - 1
            sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_c1 = model(batch["input_ids"], labels=batch["labels"])
                    loss_c1 = outputs_c1.loss

                with torch.no_grad():
                    preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
                    targets_c1 = batch["labels"][:, 1:]
                    mask_c1 = targets_c1 != -100
                    acc_c1 = (
                        (preds_c1[mask_c1] == targets_c1[mask_c1]).float().mean().item()
                        if mask_c1.any()
                        else 0.0
                    )
                    total_acc_c1 += acc_c1
                    total_loss_c1 += loss_c1.item()

                (loss_c1 * loss_scale).backward()

        # Zero out C1 gradients on keyed weights.
        # NOTE: This operates on raw_model (unwrapped), which is safe because
        # DDP's allreduce hooks fire during the final synced backward above.
        # By this point .grad tensors are already reduced across ranks, so
        # subsequent in-place modifications (masking, scaling) are purely local
        # and don't interact with DDP's communication.
        mask_keyed_gradients(raw_model, key, plan=mask_plan)

        if do_c2:
            # ==================== STEP 2: APPLY PERMUTATION (C1 -> C2) ====================
            apply_permutation(raw_model, key, plan=swap_plan)

            # ==================== STEP 3: KEYED ARCHITECTURE (C2) ====================
            for micro_idx, batch in enumerate(micro_batches):
                is_last_micro = micro_idx == grad_accum_steps - 1
                sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()

                with sync_ctx:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs_c2 = model(batch["input_ids"], labels=batch["labels"])
                        loss_c2 = outputs_c2.loss

                    with torch.no_grad():
                        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
                        targets_c2 = batch["labels"][:, 1:]
                        mask_c2 = targets_c2 != -100
                        acc_c2 = (
                            (preds_c2[mask_c2] == targets_c2[mask_c2]).float().mean().item()
                            if mask_c2.any()
                            else 0.0
                        )
                        total_acc_c2 += acc_c2
                        total_loss_c2 += loss_c2.item()

                    (loss_c2 * loss_scale).backward()
        elif want_c2_metrics:
            # C2 metrics pass for logging only (no gradients / no updates).
            # Gated by --c2_metrics_every_k; disabled by default because each
            # run of this pass costs roughly one forward (~1/3 of a fwd+bwd)
            # and erases most of the wall-time savings from increasing
            # c2_every_k. Also triggers a distinct torch.compile cache entry
            # (train-mode + no_grad) that hits the remat UserWarning on the
            # checkpointed forward.
            apply_permutation(raw_model, key, plan=swap_plan)
            with torch.no_grad():
                for batch in micro_batches:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs_c2 = model(batch["input_ids"], labels=batch["labels"])
                        loss_c2 = outputs_c2.loss

                    preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
                    targets_c2 = batch["labels"][:, 1:]
                    mask_c2 = targets_c2 != -100
                    acc_c2 = (
                        (preds_c2[mask_c2] == targets_c2[mask_c2]).float().mean().item()
                        if mask_c2.any()
                        else 0.0
                    )
                    total_acc_c2 += acc_c2
                    total_loss_c2 += loss_c2.item()
            unapply_permutation(raw_model, key, plan=swap_plan)

        # Average metrics over micro-steps
        avg_loss_c1 = total_loss_c1 / grad_accum_steps
        avg_loss_c2 = total_loss_c2 / grad_accum_steps
        avg_acc_c1 = total_acc_c1 / grad_accum_steps
        avg_acc_c2 = total_acc_c2 / grad_accum_steps

        if do_c2:
            # ── Gradient combination ──
            # Each backward pass scales loss by loss_scale = 1/grad_accum_steps, so
            # after accumulation each phase contributes grad/grad_accum_steps.
            #
            # After Phase 1 masking + Phase 2 backward, the accumulated .grad is:
            #   Public weights:  grad_c1/A + grad_c2/A   (A = grad_accum_steps)
            #   Keyed weights:   grad_c2/A               (C1 contribution was zeroed)
            #
            # scale_public_gradients(..., 0.5) then halves only the public portion:
            #   Public weights:  (grad_c1 + grad_c2) / (2A)   ← averaged over both phases
            #   Keyed weights:   grad_c2 / A                   ← unchanged, full contribution
            scale_public_gradients(raw_model, key, scale=0.5, plan=mask_plan)
            # Return to C1 frame before optimizer update so Adam moments always
            # align with stable parameter positions across all steps.
            unapply_permutation(raw_model, key, plan=swap_plan)
            swap_gradients(raw_model, key, plan=swap_plan)

        # Clip and step in C1 arrangement (both C2 and non-C2 steps).
        # On non-C2 steps, keyed positions have zero-tensor grads (no C2 backward
        # ran), so a plain optimizer.step() would apply weight decay and residual
        # momentum to them.  Use adamw_step_preserving_public to freeze them.
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        if do_c2:
            optimizer.step()
        else:
            adamw_step_preserving_public(optimizer, c1_only_update_masks)
        scheduler.step()

        global_step += 1

        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step
        step_flops = flops_step_with_c2 if do_c2 else flops_step_baseline
        cumulative_flops += step_flops
        if do_c2:
            c2_passes_cumulative += 1

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            pbar.update(1)
            postfix = {"loss_c1": f"{avg_loss_c1:.3f}"}
            if have_c2_metrics:
                postfix["loss_c2"] = f"{avg_loss_c2:.3f}"
            postfix["tok/s"] = f"{tps:,.0f}"
            pbar.set_postfix(postfix)

        # Logging
        if is_main and global_step % args.log_interval == 0:
            ppl_c1 = math.exp(min(avg_loss_c1, 100))
            lr = optimizer.param_groups[0]["lr"]

            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            samples_per_sec = effective_batch / step_elapsed if step_elapsed > 0 else 0.0
            achieved_flops_per_sec = step_flops / step_elapsed if step_elapsed > 0 else 0.0
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0
            flops_increase_pct_vs_baseline = (100.0 * c2_passes_cumulative / global_step) if global_step > 0 else 0.0

            log_dict = {
                # Task losses
                "loss_c1": avg_loss_c1,
                "loss_avg": ((avg_loss_c1 + avg_loss_c2) / 2) if do_c2 else avg_loss_c1,
                "acc_c1": avg_acc_c1,
                "acc_avg": ((avg_acc_c1 + avg_acc_c2) / 2) if do_c2 else avg_acc_c1,
                "ppl_c1": ppl_c1,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "train/step": global_step,
                "train/ran_c2": int(do_c2),
                # Timing
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                # Throughput
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/tokens_per_sec_per_gpu": tokens_per_sec / world_size,
                "perf/samples_per_sec": samples_per_sec,
                # FLOPs
                "perf/flops_per_step": step_flops,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/achieved_tflops_per_gpu": per_gpu_flops / 1e12,
                "perf/mfu": mfu,
                # Cumulative
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/c2_passes_cumulative": c2_passes_cumulative,
                "perf/flops_increase_pct_vs_baseline": flops_increase_pct_vs_baseline,
                "perf/cumulative_flops": cumulative_flops,
                "perf/cumulative_petaflops": cumulative_flops / 1e15,
            }
            # Only emit C2 metrics on steps where they were actually computed
            # (either a real C2 backward or the gated no-grad metrics pass).
            # On all other steps, let wandb render a sparse series rather than
            # logging stale zeros.
            if have_c2_metrics:
                log_dict["loss_c2"] = avg_loss_c2
                log_dict["acc_c2"] = avg_acc_c2
                log_dict["ppl_c2"] = math.exp(min(avg_loss_c2, 100))

            log_dict.update(get_gpu_memory_stats(device))
            wandb.log(log_dict)

        # Validation
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            val_metrics = evaluate(
                raw_model,
                val_dataloader,
                key,
                device,
                args.eval_steps,
                is_distributed=is_distributed,
                swap_plan=swap_plan,
            )
            if is_main:
                wandb.log({**val_metrics, "train/step": global_step})

        # Save checkpoint
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                raw_model,
                tokenizer,
                optimizer,
                save_path,
                scheduler=scheduler,
                global_step=global_step,
                wandb_run_id=wandb_run_id,
                cumulative_wall_secs=cumulative_wall_secs,
                c2_passes_cumulative=c2_passes_cumulative,
                cumulative_flops=cumulative_flops,
                data_epoch=data_epoch,
            )
            print(f"Saved checkpoint to {save_path}")

    if pbar is not None:
        pbar.close()

    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(
            raw_model,
            tokenizer,
            optimizer,
            save_path,
            scheduler=scheduler,
            global_step=global_step,
            wandb_run_id=wandb_run_id,
            cumulative_wall_secs=cumulative_wall_secs,
            c2_passes_cumulative=c2_passes_cumulative,
            cumulative_flops=cumulative_flops,
            data_epoch=data_epoch,
        )

        total_flops = cumulative_flops
        flops_increase_pct_vs_baseline = (100.0 * c2_passes_cumulative / global_step) if global_step > 0 else 0.0
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE — COMPUTE SUMMARY (for paper)")
        print(f"{'=' * 60}")
        print(f"  Steps:                    {global_step:,}")
        print(f"  C2 every K steps:         {args.c2_every_k}")
        print(f"  C2 passes executed:       {c2_passes_cumulative:,}")
        print(f"  FLOPs increase vs base:   {flops_increase_pct_vs_baseline:.2f}%")
        print(f"  Total parameters (N):     {num_params:,}")
        print(
            "  Current swapped params:   "
            f"{swappable_params['total']:,} "
            f"({swappable_pct_of_max:.2f}% of swappable, {swappable_pct_of_total:.2f}% of total params)"
        )
        print(f"    - attention:            {swappable_params['attention']:,}")
        print(f"    - mlp:                  {swappable_params['mlp']:,}")
        print(f"  Max swappable params:     {max_swappable_params['total']:,} ({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"    - attention:            {max_swappable_params['attention']:,}")
        print(f"    - mlp:                  {max_swappable_params['mlp']:,}")
        print(f"  Total tokens (D):         {cumulative_tokens:,}")
        print(f"  Total FLOPs:              {total_flops:.4e}")
        print(f"  Total PetaFLOPs:          {total_flops / 1e15:.2f}")
        print(f"  Wall clock (train):       {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):       {(time.time() - train_start_wall) / 3600:.2f} hours")
        avg_tokens_per_sec = (cumulative_tokens / cumulative_wall_secs) if cumulative_wall_secs > 0 else 0.0
        avg_tokens_per_sec_per_gpu = (
            avg_tokens_per_sec / world_size if world_size > 0 else 0.0
        )
        print(f"  Avg tokens/sec:           {avg_tokens_per_sec:,.0f}")
        print(f"  Avg tokens/sec/GPU:       {avg_tokens_per_sec_per_gpu:,.0f}")
        if gpu_peak_flops > 0 and cumulative_wall_secs > 0 and world_size > 0:
            avg_mfu = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops
            print(f"  Avg MFU:                  {avg_mfu:.2%}")
        print(f"  GPU:                      {gpu_name} x {world_size}")
        print(f"  Checkpoint:               {save_path}")
        print(f"{'=' * 60}\n")

        wandb.run.summary.update(
            {
                "final/total_steps": global_step,
                "final/total_tokens": cumulative_tokens,
                "final/total_flops": total_flops,
                "final/total_petaflops": total_flops / 1e15,
                "final/c2_every_k": args.c2_every_k,
                "final/c2_passes_cumulative": c2_passes_cumulative,
                "final/flops_increase_pct_vs_baseline": flops_increase_pct_vs_baseline,
                "final/wall_clock_hours": cumulative_wall_secs / 3600,
                "final/avg_tokens_per_sec": avg_tokens_per_sec,
                "final/avg_tokens_per_sec_per_gpu": avg_tokens_per_sec_per_gpu,
                "final/total_params": num_params,
                "final/swappable_params": swappable_params["total"],
                "final/swappable_attention_params": swappable_params["attention"],
                "final/swappable_mlp_params": swappable_params["mlp"],
                "final/swappable_pct_of_max": swappable_pct_of_max,
                "final/swappable_pct_of_total": swappable_pct_of_total,
                "final/max_swappable_params": max_swappable_params["total"],
                "final/max_swappable_attention_params": max_swappable_params["attention"],
                "final/max_swappable_mlp_params": max_swappable_params["mlp"],
                "final/max_swappable_pct_of_total": max_swappable_pct_of_total,
                "final/gpu_name": gpu_name,
                "final/num_gpus": world_size,
            }
        )
        if gpu_peak_flops > 0 and cumulative_wall_secs > 0 and world_size > 0:
            wandb.run.summary["final/avg_mfu"] = (
                total_flops / cumulative_wall_secs / world_size
            ) / gpu_peak_flops

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
