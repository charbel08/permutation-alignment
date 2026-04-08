"""N-Tiered Alignment Pretraining — Naive (all tiers every step).

Unlike the sampling script (pretrain_ntier.py) which keeps a constant
2-pass budget by sampling one keyed tier per step, this script trains
ALL keyed tiers at every step. The compute cost scales linearly:
  (1 + K) forward+backward passes per step, where K = number of keyed tiers.

This serves as an upper-bound baseline: every tier's keyed weights see
every training step, and public weights see gradients from all configs.

Each step:
1. Forward+backward on C1 (public architecture) — always
2. Forward+backward on C2, C3, ..., CN — all keyed tiers
3. Update weights:
   - S'_k (keyed weights for tier k): gradient from C_k only
   - S   (public weights): average of C1 and all C_k gradients

Gradient combination (N = 1 + K total configs):
   Public weights:  (grad_c1 + grad_c2 + ... + grad_cN) / N
   Keyed weights k: grad_ck  (only from that tier's pass)

Usage:
  torchrun --nproc_per_node=4 pretrain_ntier_naive.py \\
    --data_path ./data/tokenized \\
    --output_dir ./checkpoints \\
    --key_paths key1.json key2.json key3.json \\
    --max_steps 100000
"""

import argparse
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
from tiered.permutation import load_key, PermutationKey
from tiered.permutation.masking import mask_keyed_gradients, build_mask_plan, MaskPlan
from tiered.permutation.permute import (
    apply_permutation, unapply_permutation, swap_gradients, build_swap_plan,
    SwapPlan,
)
from tiered.permutation.utils import _get_attention_module, _get_mlp_module
from tiered.train.utils import load_model, save_checkpoint


# ---------------------------------------------------------------------------
# Compute-metrics helpers (identical to pretrain_ntier.py)
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


def estimate_flops_per_token(num_layers, hidden_size, intermediate_size,
                              context_size, vocab_size, num_params) -> dict:
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
        "approx_6N": 6 * num_params,
    }


_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12, "A10G": 70e12, "H100": 990e12, "H200": 990e12,
    "L4": 121e12, "L40": 181e12, "L40S": 366e12,
    "4090": 330e12, "3090": 142e12, "4080": 203e12,
}


def detect_gpu_peak_flops(device):
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower().replace(" ", ""):
            return peak, name
    return 0.0, name


def get_gpu_memory_stats(device):
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/mem_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/mem_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


# ---------------------------------------------------------------------------
# Public mask: precomputed per-parameter masks for efficient gradient scaling
# ---------------------------------------------------------------------------

@dataclass
class PublicMask:
    """Pre-computed boolean masks identifying public (non-keyed) positions.

    Built once at startup from the union of all tiers' keyed indices.
    Used in Phase 3 to scale only public gradient positions by 1/N,
    leaving keyed positions at their full single-tier value.

    Each entry maps id(param) -> (mask_1d, dim):
      - mask_1d: BoolTensor where True = public position
      - dim: 0 for row-indexed params (q/k/v weights, c_fc weight/bias),
             1 for column-indexed params (out_proj weight, c_proj weight)
    """
    entries: Dict[int, Tuple[torch.Tensor, int]] = field(default_factory=dict)


def build_public_mask(model, tiers: list, device: torch.device) -> PublicMask:
    """Build per-parameter boolean masks from the union of all tiers' keyed indices.

    Since keys are guaranteed non-overlapping, the union is a simple
    concatenation of each tier's keyed indices per (layer, component).

    Call ONCE at startup alongside swap/mask plans.
    """
    mask = PublicMask()

    # Collect all keyed indices per (layer, param_id) across all tiers.
    # Attention: row-indexed for q/k/v, column-indexed for out_proj.
    # MLP: row-indexed for c_fc, column-indexed for c_proj.
    keyed_attn_per_layer: Dict[int, Set[int]] = defaultdict(set)
    keyed_attn_out_per_layer: Dict[int, Set[int]] = defaultdict(set)
    keyed_mlp_per_layer: Dict[int, Set[int]] = defaultdict(set)

    for tier in tiers:
        for layer_idx, idx in tier.mask_plan.keyed_attn_indices.items():
            keyed_attn_per_layer[layer_idx].update(idx.cpu().tolist())
        for layer_idx, idx in tier.mask_plan.keyed_attn_out_indices.items():
            keyed_attn_out_per_layer[layer_idx].update(idx.cpu().tolist())
        for layer_idx, idx in tier.mask_plan.keyed_mlp_indices.items():
            keyed_mlp_per_layer[layer_idx].update(idx.cpu().tolist())

    for layer_idx in range(len(model.transformer.h)):
        attn = _get_attention_module(model, layer_idx)
        mlp = _get_mlp_module(model, layer_idx)

        # --- Attention ---
        if layer_idx in keyed_attn_per_layer:
            keyed = torch.tensor(sorted(keyed_attn_per_layer[layer_idx]),
                                 dtype=torch.long, device=device)
            num_rows = attn.q_proj.weight.shape[0]
            row_mask = torch.ones(num_rows, dtype=torch.bool, device=device)
            row_mask[keyed] = False

            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                mask.entries[id(proj.weight)] = (row_mask, 0)

        if layer_idx in keyed_attn_per_layer or layer_idx in keyed_attn_out_per_layer:
            keyed_cols = set(keyed_attn_per_layer.get(layer_idx, set()))
            keyed_cols.update(keyed_attn_out_per_layer.get(layer_idx, set()))
            keyed = torch.tensor(sorted(keyed_cols), dtype=torch.long, device=device)
            num_cols = attn.out_proj.weight.shape[1]
            col_mask = torch.ones(num_cols, dtype=torch.bool, device=device)
            col_mask[keyed] = False
            mask.entries[id(attn.out_proj.weight)] = (col_mask, 1)

        # --- MLP ---
        if layer_idx in keyed_mlp_per_layer:
            keyed = torch.tensor(sorted(keyed_mlp_per_layer[layer_idx]),
                                 dtype=torch.long, device=device)
            num_rows = mlp.c_fc.weight.shape[0]
            row_mask = torch.ones(num_rows, dtype=torch.bool, device=device)
            row_mask[keyed] = False
            mask.entries[id(mlp.c_fc.weight)] = (row_mask, 0)

            if mlp.c_fc.bias is not None:
                bias_mask = torch.ones(mlp.c_fc.bias.shape[0],
                                       dtype=torch.bool, device=device)
                bias_mask[keyed] = False
                mask.entries[id(mlp.c_fc.bias)] = (bias_mask, 0)

            num_cols = mlp.c_proj.weight.shape[1]
            col_mask = torch.ones(num_cols, dtype=torch.bool, device=device)
            col_mask[keyed] = False
            mask.entries[id(mlp.c_proj.weight)] = (col_mask, 1)

    return mask


def scale_public_gradients_multi(model, public_mask: PublicMask,
                                  scale: float) -> None:
    """Scale gradients at public positions by `scale`, leaving keyed positions
    untouched.

    Uses the precomputed PublicMask for a single efficient pass over all
    parameters. Parameters not in the mask (embeddings, layernorms, etc.)
    are entirely public and scaled uniformly.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        entry = public_mask.entries.get(id(p))
        if entry is None:
            # Entirely public parameter (embeddings, layernorms, biases not in MLP)
            p.grad.mul_(scale)
        else:
            pmask, dim = entry
            if dim == 0:
                if p.grad.dim() == 1:
                    # 1D param (bias): direct boolean index
                    p.grad[pmask] *= scale
                else:
                    # 2D param: public rows
                    p.grad[pmask] *= scale
            else:
                # 2D param: public columns
                p.grad[:, pmask] *= scale


# ---------------------------------------------------------------------------
# Gradient save/restore helpers for cross-tier isolation
# ---------------------------------------------------------------------------

def _extract_keyed_gradients(model, plan: MaskPlan) -> dict:
    """Save a copy of .grad values at keyed positions.

    Returns a nested dict: {(component, layer_idx): {param_name: tensor_clone}}.
    """
    saved = {}
    for layer_idx, idx in plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_data = {}
        for name, proj in [("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)]:
            if proj.weight.grad is not None:
                layer_data[f"{name}_rows"] = proj.weight.grad[idx].clone()
        if attn.out_proj.weight.grad is not None:
            layer_data["out_cols"] = attn.out_proj.weight.grad[:, idx].clone()
        saved[("attn", layer_idx)] = layer_data

    for layer_idx, idx in plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_data = {}
        if attn.out_proj.weight.grad is not None:
            layer_data["out_cols"] = attn.out_proj.weight.grad[:, idx].clone()
        saved[("attn_out", layer_idx)] = layer_data

    for layer_idx, idx in plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        layer_data = {}
        if mlp.c_fc.weight.grad is not None:
            layer_data["fc_rows"] = mlp.c_fc.weight.grad[idx].clone()
        if mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            layer_data["fc_bias"] = mlp.c_fc.bias.grad[idx].clone()
        if mlp.c_proj.weight.grad is not None:
            layer_data["proj_cols"] = mlp.c_proj.weight.grad[:, idx].clone()
        saved[("mlp", layer_idx)] = layer_data

    return saved


def _restore_keyed_gradients(model, plan: MaskPlan, saved: dict):
    """Write saved gradient values back into .grad at keyed positions."""
    for layer_idx, idx in plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_data = saved[("attn", layer_idx)]
        for name, proj in [("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)]:
            key = f"{name}_rows"
            if key in layer_data and proj.weight.grad is not None:
                proj.weight.grad[idx] = layer_data[key]
        if "out_cols" in layer_data and attn.out_proj.weight.grad is not None:
            attn.out_proj.weight.grad[:, idx] = layer_data["out_cols"]

    for layer_idx, idx in plan.keyed_attn_out_indices.items():
        attn = _get_attention_module(model, layer_idx)
        layer_data = saved[("attn_out", layer_idx)]
        if "out_cols" in layer_data and attn.out_proj.weight.grad is not None:
            attn.out_proj.weight.grad[:, idx] = layer_data["out_cols"]

    for layer_idx, idx in plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        layer_data = saved[("mlp", layer_idx)]
        if "fc_rows" in layer_data and mlp.c_fc.weight.grad is not None:
            mlp.c_fc.weight.grad[idx] = layer_data["fc_rows"]
        if "fc_bias" in layer_data and mlp.c_fc.bias is not None and mlp.c_fc.bias.grad is not None:
            mlp.c_fc.bias.grad[idx] = layer_data["fc_bias"]
        if "proj_cols" in layer_data and mlp.c_proj.weight.grad is not None:
            mlp.c_proj.weight.grad[:, idx] = layer_data["proj_cols"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TierInfo:
    """Pre-computed plans and metadata for a single keyed tier."""
    tier_id: int
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="N-Tiered Alignment Pretraining — Naive (all tiers every step)")

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

    # Permutation keys — one per keyed tier
    parser.add_argument("--key_paths", type=str, nargs="+", required=True)

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
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)

    # Distributed / workers
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes per rank")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_tier(model, eval_batches, key, device, is_distributed,
                  swap_plan, tier_label="c2"):
    """Evaluate C1 + one keyed config on pre-fetched batches."""
    model.eval()
    total_loss_c1 = total_loss_ck = total_acc_c1 = total_acc_ck = 0.0
    count = 0

    for batch in eval_batches:
        input_ids, labels = batch["input_ids"], batch["labels"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        acc_c1 = (preds_c1[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0

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


def evaluate_all_tiers(model, dataloader, tiers, device, num_steps, is_distributed):
    eval_batches = _prefetch_eval_batches(dataloader, device, num_steps)
    merged = {}
    for tier in tiers:
        metrics = evaluate_tier(
            model, eval_batches, tier.key, device,
            is_distributed, tier.swap_plan, tier_label=f"c{tier.tier_id}")
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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load keys and build per-tier plans ──
    num_keyed_tiers = len(args.key_paths)
    num_configs = 1 + num_keyed_tiers  # C1 + all keyed
    if is_main:
        print(f"NAIVE MODE: training ALL {num_keyed_tiers} keyed tier(s) + C1 "
              f"= {num_configs} passes per step")

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
        tier_id = i + 2
        tiers.append(TierInfo(tier_id=tier_id, key=key,
                              swap_plan=swap_plan, mask_plan=mask_plan))
        if is_main:
            print(f"  Tier C{tier_id}: key={key_path}, "
                  f"{len(key.attn_heads)} attn swaps, {len(key.mlp_cols)} MLP swaps, "
                  f"{len(swap_plan.attn_ops)} swap ops")

    # Build the public mask from the union of all tiers' keyed indices.
    # This is used in Phase 3 to scale only public gradient positions by
    # 1/N, leaving keyed positions at their full single-tier value.
    public_mask = build_public_mask(model, tiers, device)
    if is_main:
        print(f"Built public mask: {len(public_mask.entries)} parameter tensors "
              f"have mixed public/keyed positions")

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
    if args.checkpoint:
        training_state_path = os.path.join(args.checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"Checkpoint dir exists but training_state.pt not found: {training_state_path}")
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = training_state["global_step"]
        wandb_run_id = training_state.get("wandb_run_id")
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        if is_main:
            print(f"Resumed training state from step {global_step}")
            if cumulative_wall_secs > 0:
                print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    # ── Wandb ──
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

    # ── Training loop setup ──
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
        print(f"Passes per step: {num_configs} (C1 + {num_keyed_tiers} keyed)")

    # ── Compute metrics ──
    num_params = count_total_parameters(raw_model)
    num_trainable = count_trainable_parameters(raw_model)
    swappable_params = count_swappable_parameters(raw_model, tiers[0].mask_plan)
    max_swappable_params = count_max_swappable_parameters(raw_model)
    swappable_pct_of_max = 100.0 * swappable_params["total"] / max_swappable_params["total"]
    max_swappable_pct_of_total = 100.0 * max_swappable_params["total"] / num_params

    inter_size = args.intermediate_size or (args.hidden_size * 4)
    vocab_size = raw_model.get_input_embeddings().weight.shape[0]
    flop_info = estimate_flops_per_token(
        args.num_layers, args.hidden_size, inter_size,
        args.context_size, vocab_size, num_params)

    tokens_per_step = effective_batch * args.context_size
    flops_per_token = 6 * num_params
    # num_configs passes per step (not 2 like the sampling script)
    flops_per_step = num_configs * flops_per_token * tokens_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Total parameters:           {num_params:,}")
        print(f"  Trainable parameters:       {num_trainable:,}")
        print(f"  Current swappable params:   {swappable_params['total']:,} "
              f"({swappable_pct_of_max:.2f}% of max swappable)")
        print(f"    - attention:              {swappable_params['attention']:,}")
        print(f"    - mlp:                    {swappable_params['mlp']:,}")
        print(f"  Max swappable params:       {max_swappable_params['total']:,} "
              f"({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"  Tokens/step:                {tokens_per_step:,}")
        print(f"  FLOPs/token (6N):           {flops_per_token:.3e}")
        print(f"  FLOPs/step (est):           {flops_per_step:.3e}  "
              f"({num_configs} passes × 6N × tokens)")
        print(f"  GPU:                        {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:              {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:              unknown (MFU will be N/A)")
        print()

    cumulative_tokens = global_step * tokens_per_step
    train_start_wall = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        wandb.config.update({
            "compute/num_params": num_params,
            "compute/num_trainable_params": num_trainable,
            "compute/swappable_params": swappable_params["total"],
            "compute/swappable_attention_params": swappable_params["attention"],
            "compute/swappable_mlp_params": swappable_params["mlp"],
            "compute/swappable_pct_of_max": swappable_pct_of_max,
            "compute/max_swappable_params": max_swappable_params["total"],
            "compute/tokens_per_step": tokens_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/flops_per_token_6N": flops_per_token,
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "compute/num_keyed_tiers": num_keyed_tiers,
            "compute/num_configs": num_configs,
            "compute/mode": "naive_all_tiers",
            "compute/vocab_size": vocab_size,
        }, allow_val_change=True)

    # Initial validation
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate_all_tiers(
            raw_model, val_dataloader, tiers, device,
            args.eval_steps, is_distributed)
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})

    pbar = tqdm(total=args.max_steps, desc="Training (naive)",
                initial=global_step) if is_main else None

    while global_step < args.max_steps:
        optimizer.zero_grad()
        model.train()

        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

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

        # Per-tier loss/acc tracking
        total_loss_c1 = 0.0
        total_acc_c1 = 0.0
        tier_losses = {t.tier_id: 0.0 for t in tiers}
        tier_accs = {t.tier_id: 0.0 for t in tiers}

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

        # Zero C1 gradients on ALL tiers' keyed weights so that only public
        # positions retain the C1 contribution going into Phase 2.
        for tier in tiers:
            mask_keyed_gradients(raw_model, tier.key, plan=tier.mask_plan)

        # ==================== PHASE 2: ALL keyed tiers sequentially ====================
        # To prevent cross-tier contamination (tier k's backward depositing
        # gradients on tier j's keyed positions), we save each tier's keyed
        # gradient after its pass and mask ALL keyed positions before the
        # next tier runs. After all tiers, we restore the saved gradients.
        saved_keyed_grads = {}

        for tier in tiers:
            apply_permutation(raw_model, tier.key, plan=tier.swap_plan)

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
                        tier_accs[tier.tier_id] += acc
                        tier_losses[tier.tier_id] += loss_ck.item()
                    (loss_ck * loss_scale).backward()

            # Return to C1 frame and move gradients to match
            unapply_permutation(raw_model, tier.key, plan=tier.swap_plan)
            swap_gradients(raw_model, tier.key, plan=tier.swap_plan)

            # Save this tier's keyed gradient before it gets zeroed.
            # Since all keyed positions were zeroed before this tier ran,
            # the gradient here is purely from this tier's own backward pass.
            saved_keyed_grads[tier.tier_id] = _extract_keyed_gradients(
                raw_model, tier.mask_plan)

            # Mask ALL tiers' keyed positions to remove this tier's
            # contaminating gradient on other tiers' keyed weights.
            # Public positions accumulate across all tiers (desired).
            for t in tiers:
                mask_keyed_gradients(raw_model, t.key, plan=t.mask_plan)

        # ==================== PHASE 3: gradient combination ====================
        # At this point .grad contains:
        #   Public positions:  grad_c1 + grad_c2 + ... + grad_cN  (N terms)
        #   Keyed positions:   0  (all zeroed by masking after last tier)
        #
        # We want:
        #   Public:  (grad_c1 + sum(grad_ck)) / (N * A)
        #   Keyed-k: grad_ck / A
        #
        # Step 1: Restore each tier's saved keyed gradients.
        for tier in tiers:
            _restore_keyed_gradients(
                raw_model, tier.mask_plan, saved_keyed_grads[tier.tier_id])

        # Step 2: Scale only public positions by 1/N using the precomputed
        # mask. Keyed positions already hold the correct unattenuated value.
        scale_public_gradients_multi(raw_model, public_mask,
                                      scale=1.0 / num_configs)

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
        avg_acc_c1 = total_acc_c1 / grad_accum_steps
        avg_tier_losses = {k: v / grad_accum_steps for k, v in tier_losses.items()}
        avg_tier_accs = {k: v / grad_accum_steps for k, v in tier_accs.items()}
        loss_all = [avg_loss_c1] + list(avg_tier_losses.values())
        loss_avg = sum(loss_all) / len(loss_all)

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            postfix = {"c1": f"{avg_loss_c1:.3f}"}
            for tier in tiers:
                postfix[f"c{tier.tier_id}"] = f"{avg_tier_losses[tier.tier_id]:.3f}"
            postfix["tok/s"] = f"{tps:,.0f}"
            pbar.update(1)
            pbar.set_postfix(postfix)

        if is_main and global_step % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            samples_per_sec = effective_batch / step_elapsed if step_elapsed > 0 else 0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                "loss_c1": avg_loss_c1,
                "loss_avg": loss_avg,
                "acc_c1": avg_acc_c1,
                "ppl_c1": math.exp(min(avg_loss_c1, 100)),
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "train/step": global_step,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/tokens_per_sec_per_gpu": tokens_per_sec / world_size,
                "perf/samples_per_sec": samples_per_sec,
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/achieved_tflops_per_gpu": per_gpu_flops / 1e12,
                "perf/mfu": mfu,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": num_configs * flops_per_token * cumulative_tokens,
                "perf/cumulative_petaflops": (num_configs * flops_per_token * cumulative_tokens) / 1e15,
            }
            for tier in tiers:
                tid = tier.tier_id
                log_dict[f"loss_c{tid}"] = avg_tier_losses[tid]
                log_dict[f"acc_c{tid}"] = avg_tier_accs[tid]
                log_dict[f"ppl_c{tid}"] = math.exp(min(avg_tier_losses[tid], 100))
            log_dict.update(get_gpu_memory_stats(device))
            wandb.log(log_dict)

        # ── Validation (always all tiers) ──
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            val_metrics = evaluate_all_tiers(
                raw_model, val_dataloader, tiers, device,
                args.eval_steps, is_distributed)
            if is_main:
                wandb.log({**val_metrics, "train/step": global_step})

        # ── Checkpoint ──
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                raw_model, tokenizer, optimizer, save_path,
                scheduler=scheduler, global_step=global_step,
                wandb_run_id=wandb_run_id,
                cumulative_wall_secs=cumulative_wall_secs)
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
            cumulative_wall_secs=cumulative_wall_secs)

        total_flops = num_configs * flops_per_token * cumulative_tokens
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — COMPUTE SUMMARY (naive all-tiers)")
        print(f"{'='*60}")
        print(f"  Mode:                  naive (all {num_keyed_tiers} tiers every step)")
        print(f"  Passes/step:           {num_configs}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Current swappable:     {swappable_params['total']:,} "
              f"({swappable_pct_of_max:.2f}% of max swappable)")
        print(f"    - attention:         {swappable_params['attention']:,}")
        print(f"    - mlp:               {swappable_params['mlp']:,}")
        print(f"  Max swappable:         {max_swappable_params['total']:,} "
              f"({max_swappable_pct_of_total:.2f}% of total params)")
        print(f"  Total tokens (D):      {cumulative_tokens:,}")
        print(f"  Total FLOPs ({num_configs}×6ND):  {total_flops:.4e}")
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
        print(f"{'='*60}\n")

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
            "final/max_swappable_params": max_swappable_params["total"],
            "final/gpu_name": gpu_name,
            "final/num_gpus": world_size,
            "final/num_configs": num_configs,
            "final/mode": "naive_all_tiers",
        })
        if gpu_peak_flops > 0:
            wandb.run.summary["final/avg_mfu"] = (
                total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
