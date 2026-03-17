"""PEFT LoRA private finetuning baseline for 2-tier comparison.

This script treats:
  - C1: base model (adapter disabled)
  - C2: base model + PEFT LoRA adapter (adapter enabled)

It chooses LoRA rank automatically from a parameter budget:
  budget = number of keyed parameters induced by --key_path

The selected rank is the highest integer that fits this budget for the
target GPT-Neo linear modules (q/k/v/o and MLP c_fc/c_proj).

Outputs include:
  - C1/C2 private and retain validation metrics
  - Throughput metrics
  - FLOPs estimates for LoRA and a 2-pass tiered reference
  - PEFT adapter checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "PEFT is required for lora_private_finetune.py. Install with: pip install peft"
    ) from exc

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key
from tiered.permutation.masking import build_mask_plan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj")


@dataclass
class LoRATarget:
    """Metadata for one target linear module."""

    name: str
    in_features: int
    out_features: int


@dataclass
class RankSelection:
    """Rank/budget metadata for reporting."""

    target_keyed_params: int
    lora_params_per_rank: int
    max_effective_rank: int
    selected_rank: int
    selected_lora_params: int
    budget_gap: int


def parse_args():
    parser = argparse.ArgumentParser(description="PEFT LoRA private finetuning baseline")

    # Model + budget reference
    parser.add_argument("--checkpoint", type=str, required=True, help="Base checkpoint path")
    parser.add_argument("--key_path", type=str, required=True, help="Key used only to define param budget")

    # Data
    parser.add_argument("--private_data", type=str, required=True, help="Path to private tokenized dataset")
    parser.add_argument("--public_data", type=str, default=None, help="Optional retain dataset for validation")
    parser.add_argument("--output_dir", type=str, required=True)

    # LoRA
    parser.add_argument("--rank_override", type=int, default=None, help="Manually set LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=None, help="Defaults to selected rank")
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval/logging
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-lora")
    parser.add_argument("--run_name", type=str, default=None)

    # Runtime
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_keyed_parameters(model, key, device) -> int:
    """Exact keyed-parameter count matching mask_public_gradients behavior.

    Keyed params are those whose gradients are preserved (not zeroed) by
    mask_public_gradients in tiered/permutation/masking.py:
      - Attention: q/k/v weight rows + out_proj weight cols (biases are PUBLIC)
      - MLP: c_fc weight rows + c_fc bias + c_proj weight cols (c_proj bias is PUBLIC)

    Attention biases are NOT swapped by apply_permutation and NOT preserved
    by mask_public_gradients, so they must not be counted here.
    """
    mask_plan = build_mask_plan(model, key, device)

    total_attn = 0
    total_mlp = 0

    for layer_idx, idx in mask_plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        n_idx = int(idx.numel())
        # q/k/v weight rows (keyed head indices × input dim)
        total_attn += n_idx * attn.q_proj.weight.shape[1]
        total_attn += n_idx * attn.k_proj.weight.shape[1]
        total_attn += n_idx * attn.v_proj.weight.shape[1]
        # out_proj weight cols (output dim × keyed head indices)
        total_attn += attn.out_proj.weight.shape[0] * n_idx
        # NOTE: attention biases are NOT keyed (not swapped, not masked)

    for layer_idx, idx in mask_plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        n_idx = int(idx.numel())
        # c_fc weight rows
        total_mlp += n_idx * mlp.c_fc.weight.shape[1]
        # c_fc bias (keyed — swapped and preserved by mask_public_gradients)
        if mlp.c_fc.bias is not None:
            total_mlp += n_idx
        # c_proj weight cols
        total_mlp += mlp.c_proj.weight.shape[0] * n_idx
        # NOTE: c_proj bias is NOT keyed

    return total_attn + total_mlp


def find_lora_targets(model, target_modules=TARGET_MODULES) -> list[LoRATarget]:
    """Collect linear modules PEFT will LoRA-wrap."""
    targets: list[LoRATarget] = []
    module_set = set(target_modules)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] in module_set:
            targets.append(
                LoRATarget(
                    name=name,
                    in_features=module.in_features,
                    out_features=module.out_features,
                )
            )
    targets.sort(key=lambda t: t.name)
    return targets


def lora_params_per_rank(targets: list[LoRATarget]) -> int:
    return sum(t.in_features + t.out_features for t in targets)


def max_effective_rank(targets: list[LoRATarget]) -> int:
    vals = [min(t.in_features, t.out_features) for t in targets]
    return min(vals) if vals else 0


def resolve_rank_from_budget(
    target_param_budget: int,
    per_rank_params: int,
    rank_cap: int | None = None,
) -> tuple[int, int]:
    if per_rank_params <= 0:
        raise ValueError("per_rank_params must be > 0")
    if target_param_budget <= 0:
        raise ValueError("target_param_budget must be > 0")

    rank = target_param_budget // per_rank_params
    if rank < 1:
        rank = 1
    if rank_cap is not None:
        rank = min(rank, rank_cap)
    return rank, rank * per_rank_params


def build_rank_selection(model, key, device, rank_override: int | None) -> tuple[RankSelection, list[LoRATarget]]:
    target_keyed = count_keyed_parameters(model, key, device)
    targets = find_lora_targets(model, target_modules=TARGET_MODULES)
    per_rank = lora_params_per_rank(targets)
    rank_cap = max_effective_rank(targets)

    if rank_override is not None:
        rank = rank_override
        if rank_cap > 0:
            rank = min(rank, rank_cap)
        selected_params = rank * per_rank
    else:
        rank, selected_params = resolve_rank_from_budget(
            target_param_budget=target_keyed,
            per_rank_params=per_rank,
            rank_cap=rank_cap if rank_cap > 0 else None,
        )

    return (
        RankSelection(
            target_keyed_params=target_keyed,
            lora_params_per_rank=per_rank,
            max_effective_rank=rank_cap,
            selected_rank=rank,
            selected_lora_params=selected_params,
            budget_gap=target_keyed - selected_params,
        ),
        targets,
    )


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
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
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    normalized = name.lower().replace(" ", "")
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in normalized:
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


def get_train_val_splits(dataset):
    if "train" in dataset and "test" in dataset:
        return dataset["train"], dataset["test"]
    if "train" in dataset:
        train = dataset["train"]
        val = train.select(range(min(1000, len(train))))
        return train, val

    n_val = max(100, len(dataset) // 10)
    train = dataset.select(range(len(dataset) - n_val))
    val = dataset.select(range(len(dataset) - n_val, len(dataset)))
    return train, val


def filter_columns(ds):
    keep = {"input_ids", "attention_mask"}
    to_remove = [c for c in ds.column_names if c not in keep]
    if to_remove:
        ds = ds.remove_columns(to_remove)
    return ds


def token_metrics(logits, labels):
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any():
        return 0.0, 0.0

    top1 = (preds[mask] == targets[mask]).float().mean().item()
    top3_idx = logits[:, :-1, :].topk(3, dim=-1).indices
    top3 = (top3_idx[mask] == targets[mask].unsqueeze(-1)).any(dim=-1).float().mean().item()
    return top1, top3


@contextmanager
def adapters_disabled(model):
    """Compatibility shim across PEFT versions."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return

    # Fallback for older APIs
    if hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters"):
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
        return

    raise AttributeError("Could not find adapter disable API on this PEFT model")


@torch.no_grad()
def evaluate_c1_c2(model, dataloader, device, num_steps: int = 50):
    """Evaluate the same batches with adapter off (C1) and on (C2)."""
    was_training = model.training
    model.eval()

    total_loss_c1 = total_acc_c1 = total_top3_c1 = 0.0
    total_loss_c2 = total_acc_c2 = total_top3_c2 = 0.0
    n = 0

    data_iter = iter(dataloader)
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                batch = next(data_iter)
            except StopIteration:
                break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        with adapters_disabled(model):
            with autocast_ctx:
                out_c1 = model(input_ids, labels=labels)
        acc1, top31 = token_metrics(out_c1.logits, labels)

        with autocast_ctx:
            out_c2 = model(input_ids, labels=labels)
        acc2, top32 = token_metrics(out_c2.logits, labels)

        total_loss_c1 += out_c1.loss.item()
        total_acc_c1 += acc1
        total_top3_c1 += top31
        total_loss_c2 += out_c2.loss.item()
        total_acc_c2 += acc2
        total_top3_c2 += top32
        n += 1

    # Restore original training state
    if was_training:
        model.train()

    if n == 0:
        return {
            "loss_c1": float("nan"),
            "ppl_c1": float("nan"),
            "acc_c1": float("nan"),
            "top3_c1": float("nan"),
            "loss_c2": float("nan"),
            "ppl_c2": float("nan"),
            "acc_c2": float("nan"),
            "top3_c2": float("nan"),
        }

    avg_loss_c1 = total_loss_c1 / n
    avg_loss_c2 = total_loss_c2 / n
    return {
        "loss_c1": avg_loss_c1,
        "ppl_c1": math.exp(min(avg_loss_c1, 100)),
        "acc_c1": total_acc_c1 / n,
        "top3_c1": total_top3_c1 / n,
        "loss_c2": avg_loss_c2,
        "ppl_c2": math.exp(min(avg_loss_c2, 100)),
        "acc_c2": total_acc_c2 / n,
        "top3_c2": total_top3_c2 / n,
    }


def save_adapter_checkpoint(
    path: str,
    raw_model,
    optimizer,
    scheduler,
    global_step: int,
    cumulative_wall_secs: float,
    wandb_run_id: str | None,
    metadata: dict,
):
    os.makedirs(path, exist_ok=True)

    raw_model.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "cumulative_wall_secs": cumulative_wall_secs,
            "wandb_run_id": wandb_run_id,
        },
        os.path.join(path, "training_state.pt"),
    )
    with open(os.path.join(path, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_tokenizer(checkpoint_path: str):
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
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
    is_distributed = local_rank != -1

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model and key for budgeting
    base_model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    base_model.to(device)
    key = load_key(args.key_path)

    # ── FIX #1: capture base param count BEFORE PEFT wrapping ──
    # After get_peft_model, .parameters() includes LoRA A/B matrices,
    # which would inflate N and skew all FLOPs estimates.
    total_base_params = count_total_parameters(base_model)

    # Determine LoRA rank from keyed-param budget
    rank_meta, targets = build_rank_selection(base_model, key, device, args.rank_override)
    alpha = args.lora_alpha if args.lora_alpha is not None else float(rank_meta.selected_rank)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank_meta.selected_rank,
        lora_alpha=alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=list(TARGET_MODULES),
    )
    raw_model = get_peft_model(base_model, peft_config)
    raw_model.to(device)

    trainable_lora_params = count_trainable_parameters(raw_model)
    if trainable_lora_params != rank_meta.selected_lora_params:
        rank_meta.selected_lora_params = trainable_lora_params
        rank_meta.budget_gap = rank_meta.target_keyed_params - trainable_lora_params

    if is_main:
        print(f"Loaded base checkpoint: {args.checkpoint}")
        print(f"Budget key: {args.key_path}")
        print(f"Budget (keyed params):      {rank_meta.target_keyed_params:,}")
        print(f"LoRA params per rank:       {rank_meta.lora_params_per_rank:,}")
        print(f"Max effective rank:         {rank_meta.max_effective_rank}")
        print(f"Selected LoRA rank:         {rank_meta.selected_rank}")
        print(f"Selected LoRA params:       {rank_meta.selected_lora_params:,}")
        print(f"Budget gap (keyed - LoRA):  {rank_meta.budget_gap:,}")
        print(f"Number of LoRA targets:     {len(targets)}")
        print(f"LoRA alpha:                 {alpha}")
        print(f"LoRA dropout:               {args.lora_dropout}")
        print(f"Base model params (N):      {total_base_params:,}")
        print(f"Total params (N + LoRA):    {count_total_parameters(raw_model):,}")
        raw_model.print_trainable_parameters()

    # Wrap for DDP
    model = raw_model
    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank])
        if is_main:
            print(f"DDP enabled on {world_size} GPUs")

    # Data
    tokenizer = load_tokenizer(args.checkpoint)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    private_ds = load_from_disk(args.private_data)
    private_train, private_val = get_train_val_splits(private_ds)
    private_train = filter_columns(private_train)
    private_val = filter_columns(private_val)

    private_sampler = DistributedSampler(private_train, shuffle=True) if is_distributed else None
    private_loader = DataLoader(
        private_train,
        batch_size=args.batch_size,
        sampler=private_sampler,
        shuffle=(private_sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    private_val_loader = DataLoader(
        private_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    retain_val_loader = None
    if args.public_data is not None:
        public_ds = load_from_disk(args.public_data)
        _, retain_val = get_train_val_splits(public_ds)
        retain_val = filter_columns(retain_val)
        retain_val_loader = DataLoader(
            retain_val,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Optimizer on trainable LoRA params only
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95))

    warmup_steps = min(args.warmup_steps, max(args.max_steps - 1, 0))
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(warmup_steps, 1),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.max_steps - warmup_steps, 1),
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ── FIX #1 continued: use total_base_params (N) for all FLOPs math ──
    context_size = raw_model.config.max_position_embeddings
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    tokens_per_step = effective_batch * context_size
    tokens_per_microbatch = args.batch_size * context_size * world_size
    tokens_private_per_step = tokens_per_step
    tokens_public_per_step = 0

    # FLOPs estimates (N = base params, P = LoRA params):
    # - lora_full_equiv: treat as full fwd+bwd over base params (6N/token)
    # - lora_approx: frozen-base training estimate (4N + 6P_lora)/token
    # - tiered_2pass_ref: 2x full passes (12N/token)
    flops_lora_full_equiv = 6 * total_base_params * tokens_per_step
    flops_lora_approx = (4 * total_base_params + 6 * rank_meta.selected_lora_params) * tokens_per_step
    flops_tiered_2pass_ref = 12 * total_base_params * tokens_per_step
    # Primary FLOPs line for graph parity with private_finetune.py
    flops_per_step = flops_lora_approx

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        print("\n── Compute metrics ──")
        print(f"  Base parameters (N):   {total_base_params:,}")
        print(f"  LoRA parameters:       {rank_meta.selected_lora_params:,}")
        print(f"  Context size:          {context_size}")
        print(f"  World size:            {world_size}")
        print(f"  Grad accum steps:      {args.grad_accum_steps}")
        print(f"  Effective batch size:  {effective_batch}")
        print(f"  Tokens/step (private): {tokens_private_per_step:,}")
        print(f"  Tokens/step (public):  {tokens_public_per_step:,}")
        print(f"  FLOPs/step (primary):  {flops_per_step:.3e}  (LoRA approx)")
        print(f"  FLOPs/step (full-eq):  {flops_lora_full_equiv:.3e}")
        print(f"  FLOPs/step (tiered):   {flops_tiered_2pass_ref:.3e}")
        print(f"  GPU:                   {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:         {gpu_peak_flops:.3e} FLOP/s")
        else:
            print("  GPU peak bf16:         unknown (MFU will be N/A)")
        print()

    run_metadata = {
        "base_checkpoint": args.checkpoint,
        "budget_key_path": args.key_path,
        "private_data": args.private_data,
        "public_data": args.public_data,
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch,
        "context_size": context_size,
        "tokens_per_step_global": tokens_per_step,
        "tokens_private_per_step": tokens_private_per_step,
        "tokens_public_per_step": tokens_public_per_step,
        "total_base_params": total_base_params,
        "rank_selection": asdict(rank_meta),
        "target_modules": list(TARGET_MODULES),
        "lora_alpha": alpha,
        "lora_dropout": args.lora_dropout,
        "num_lora_targets": len(targets),
        "lora_targets": [asdict(t) for t in targets],
        "flops_estimates": {
            "primary_per_step": flops_per_step,
            "lora_full_equiv_per_step": flops_lora_full_equiv,
            "lora_approx_per_step": flops_lora_approx,
            "tiered_2pass_ref_per_step": flops_tiered_2pass_ref,
            "tiered_vs_lora_full_equiv_ratio": (
                flops_tiered_2pass_ref / flops_lora_full_equiv if flops_lora_full_equiv > 0 else float("nan")
            ),
            "tiered_vs_lora_approx_ratio": (
                flops_tiered_2pass_ref / flops_lora_approx if flops_lora_approx > 0 else float("nan")
            ),
        },
    }

    # W&B
    wandb_run_id = None
    if is_main:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        wandb.config.update(
            {
                "compute/num_params": total_base_params,
                "compute/num_params_total": total_base_params,
                "compute/num_params_lora": rank_meta.selected_lora_params,
                "compute/context_size": context_size,
                "compute/world_size": world_size,
                "compute/grad_accum_steps": args.grad_accum_steps,
                "compute/effective_batch_size": effective_batch,
                "compute/tokens_per_step": tokens_per_step,
                "compute/tokens_private_per_step": tokens_private_per_step,
                "compute/tokens_public_per_step": tokens_public_per_step,
                "compute/flops_per_step": flops_per_step,
                "compute/flops_lora_full_equiv_per_step": flops_lora_full_equiv,
                "compute/flops_lora_approx_per_step": flops_lora_approx,
                "compute/flops_tiered_2pass_ref_per_step": flops_tiered_2pass_ref,
                "compute/kl_enabled": False,
                "compute/gpu_name": gpu_name,
                "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            },
            allow_val_change=True,
        )

    # Training loop
    if is_main:
        print(f"Starting LoRA finetuning for {args.max_steps} steps")
        print("Objective: L_ft = L_priv(C2) [LoRA adapter, no KL term]")
        print(f"Validation every {args.eval_interval} steps")
        print("Tracking: C1 on retain, C1 on private, C2 on private")

    private_iter = iter(private_loader)
    global_step = 0
    cumulative_wall_secs = 0.0
    cumulative_tokens = 0
    best_private_c2 = float("inf")
    final_metrics = {}
    train_start_wall = time.time()

    pbar = tqdm(total=args.max_steps, desc="LoRA Finetune", disable=not is_main)

    while global_step < args.max_steps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        optimizer.zero_grad(set_to_none=True)

        # ── FIX #6: gradient accumulation loop ──
        accum_loss = 0.0
        accum_acc = 0.0
        accum_top3 = 0.0

        for micro_step in range(args.grad_accum_steps):
            try:
                batch = next(private_iter)
            except StopIteration:
                if private_sampler is not None:
                    private_sampler.set_epoch(global_step + 1)
                private_iter = iter(private_loader)
                batch = next(private_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            model.train()

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )

            # For DDP: only sync gradients on the last micro-step
            no_sync_ctx = (
                model.no_sync()
                if is_distributed and micro_step < args.grad_accum_steps - 1
                else nullcontext()
            )

            with no_sync_ctx:
                with autocast_ctx:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss / args.grad_accum_steps

                loss.backward()

            # ── FIX #7: only compute token_metrics during eval or logging steps ──
            # Avoids large topk allocation during every micro-step
            if is_main and global_step % args.log_interval == (args.log_interval - 1):
                with torch.no_grad():
                    acc, top3 = token_metrics(outputs.logits, labels)
                accum_acc += acc / args.grad_accum_steps
                accum_top3 += top3 / args.grad_accum_steps

            accum_loss += outputs.loss.item() / args.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step

        if is_main:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{accum_loss:.3f}"})

            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
                achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0.0
                tflops_full = flops_lora_full_equiv / step_elapsed / 1e12 if step_elapsed > 0 else 0.0
                tflops_approx = flops_lora_approx / step_elapsed / 1e12 if step_elapsed > 0 else 0.0
                mfu = achieved_flops_per_sec / gpu_peak_flops if gpu_peak_flops > 0 else 0.0
                ppl = math.exp(min(accum_loss, 100))

                log_dict = {
                    "Train/Total Loss": accum_loss,
                    "Train/Private Loss (C2)": accum_loss,
                    "Train/KL Divergence": 0.0,
                    "Train/Perplexity (C2)": ppl,
                    "Train/Accuracy (C2)": accum_acc,
                    "Train/Top3 C2": accum_top3,
                    "Train/LR": lr,
                    "Train/Grad Norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                    "train/step": global_step,
                    # Timing
                    "perf/step_time_sec": step_elapsed,
                    "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                    "perf/wall_since_launch_hrs": (time.time() - train_start_wall) / 3600,
                    # Throughput
                    "perf/tokens_per_sec": tokens_per_sec,
                    # FLOPs
                    "perf/flops_per_step": flops_per_step,
                    "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                    "perf/mfu": mfu,
                    "perf/flops_per_step_lora_full_equiv": flops_lora_full_equiv,
                    "perf/flops_per_step_lora_approx": flops_lora_approx,
                    "perf/flops_per_step_tiered_2pass_ref": flops_tiered_2pass_ref,
                    "perf/achieved_tflops_lora_full_equiv": tflops_full,
                    "perf/achieved_tflops_lora_approx": tflops_approx,
                    # Cumulative
                    "perf/cumulative_tokens": cumulative_tokens,
                    "perf/cumulative_flops": flops_per_step * global_step,
                    "perf/cumulative_petaflops": (flops_per_step * global_step) / 1e15,
                }
                log_dict.update(get_gpu_memory_stats(device))
                wandb.log(log_dict)

        if is_main and (global_step == 1 or global_step % args.eval_interval == 0):
            print(f"\n[Validation @ step {global_step}]")
            metrics_private = evaluate_c1_c2(raw_model, private_val_loader, device, num_steps=args.eval_steps)
            print(
                f"  Private: C1 loss={metrics_private['loss_c1']:.4f} acc={metrics_private['acc_c1']:.4f} | "
                f"C2 loss={metrics_private['loss_c2']:.4f} acc={metrics_private['acc_c2']:.4f}"
            )

            eval_log = {
                "train/step": global_step,
                "Val Private/C1 Loss": metrics_private["loss_c1"],
                "Val Private/C1 Perplexity": metrics_private["ppl_c1"],
                "Val Private/C1 Accuracy": metrics_private["acc_c1"],
                "Val Private/C1 Top3": metrics_private["top3_c1"],
                "Val Private/C2 Loss": metrics_private["loss_c2"],
                "Val Private/C2 Perplexity": metrics_private["ppl_c2"],
                "Val Private/C2 Accuracy": metrics_private["acc_c2"],
                "Val Private/C2 Top3": metrics_private["top3_c2"],
            }

            metrics_retain = None
            if retain_val_loader is not None:
                metrics_retain = evaluate_c1_c2(raw_model, retain_val_loader, device, num_steps=args.eval_steps)
                print(
                    f"  Retain:  C1 loss={metrics_retain['loss_c1']:.4f} acc={metrics_retain['acc_c1']:.4f} | "
                    f"C2 loss={metrics_retain['loss_c2']:.4f} acc={metrics_retain['acc_c2']:.4f}"
                )
                eval_log.update(
                    {
                        "Val Retain/C1 Loss": metrics_retain["loss_c1"],
                        "Val Retain/C1 Perplexity": metrics_retain["ppl_c1"],
                        "Val Retain/C1 Accuracy": metrics_retain["acc_c1"],
                        "Val Retain/C1 Top3": metrics_retain["top3_c1"],
                        "Val Retain/C2 Loss": metrics_retain["loss_c2"],
                        "Val Retain/C2 Perplexity": metrics_retain["ppl_c2"],
                        "Val Retain/C2 Accuracy": metrics_retain["acc_c2"],
                        "Val Retain/C2 Top3": metrics_retain["top3_c2"],
                    }
                )

            wandb.log(eval_log)

            final_metrics = {"private": metrics_private}
            if metrics_retain is not None:
                final_metrics["retain"] = metrics_retain

            if metrics_private["loss_c2"] < best_private_c2:
                best_private_c2 = metrics_private["loss_c2"]
                save_adapter_checkpoint(
                    path=os.path.join(args.output_dir, "best"),
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    cumulative_wall_secs=cumulative_wall_secs,
                    wandb_run_id=wandb_run_id,
                    metadata=run_metadata,
                )
                print(f"  New best checkpoint saved (private C2 loss={best_private_c2:.4f})")

        if is_main and global_step % args.save_interval == 0:
            save_adapter_checkpoint(
                path=os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                raw_model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                cumulative_wall_secs=cumulative_wall_secs,
                wandb_run_id=wandb_run_id,
                metadata=run_metadata,
            )

    if is_main:
        pbar.close()

        final_dir = os.path.join(args.output_dir, "final")
        save_adapter_checkpoint(
            path=final_dir,
            raw_model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            cumulative_wall_secs=cumulative_wall_secs,
            wandb_run_id=wandb_run_id,
            metadata=run_metadata,
        )

        total_flops = flops_per_step * global_step
        total_flops_lora_full = flops_lora_full_equiv * global_step
        total_flops_lora_approx = flops_lora_approx * global_step
        total_flops_tiered_ref = flops_tiered_2pass_ref * global_step

        comparison = {
            "steps": global_step,
            "cumulative_tokens": cumulative_tokens,
            "wall_clock_hours": cumulative_wall_secs / 3600.0,
            "rank_selection": asdict(rank_meta),
            "flops_totals": {
                "primary": total_flops,
                "lora_full_equiv": total_flops_lora_full,
                "lora_approx": total_flops_lora_approx,
                "tiered_2pass_ref": total_flops_tiered_ref,
                "tiered_vs_lora_full_equiv_ratio": (
                    total_flops_tiered_ref / total_flops_lora_full if total_flops_lora_full > 0 else float("nan")
                ),
                "tiered_vs_lora_approx_ratio": (
                    total_flops_tiered_ref / total_flops_lora_approx if total_flops_lora_approx > 0 else float("nan")
                ),
            },
            "final_eval": final_metrics,
        }

        with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n{'='*60}")
        print("FINETUNING COMPLETE — COMPUTE SUMMARY")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Base parameters (N):   {total_base_params:,}")
        print(f"  LoRA params:           {rank_meta.selected_lora_params:,}")
        print(f"  World size:            {world_size}")
        print(f"  Grad accum steps:      {args.grad_accum_steps}")
        print(f"  Effective batch size:  {effective_batch}")
        print("  KL enabled:            False")
        print(f"  Total tokens:          {cumulative_tokens:,}")
        print(f"  Total FLOPs:           {total_flops:.4e}")
        print(f"  Total PetaFLOPs:       {total_flops / 1e15:.2f}")
        print(f"  Total FLOPs (full-eq): {total_flops_lora_full:.4e}")
        print(f"  Total FLOPs (tiered):  {total_flops_tiered_ref:.4e}")
        print(f"  Wall clock (train):    {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):    {(time.time() - train_start_wall) / 3600:.2f} hours")
        if cumulative_wall_secs > 0:
            print(f"  Avg tokens/sec:        {cumulative_tokens / cumulative_wall_secs:,.0f}")
            if gpu_peak_flops > 0:
                avg_mfu = (total_flops / cumulative_wall_secs) / gpu_peak_flops
                print(f"  Avg MFU:               {avg_mfu:.2%}")
        print(f"  GPU:                   {gpu_name}")
        print(f"  Checkpoint:            {final_dir}")
        print(f"  LoRA rank:             {rank_meta.selected_rank}")
        print(f"  Keyed-param budget:    {rank_meta.target_keyed_params:,}")
        ratio_full = comparison["flops_totals"]["tiered_vs_lora_full_equiv_ratio"]
        ratio_approx = comparison["flops_totals"]["tiered_vs_lora_approx_ratio"]
        print(f"  FLOPs ratio tiered/full-eq: {ratio_full:.2f}x")
        print(f"  FLOPs ratio tiered/approx:  {ratio_approx:.2f}x")
        print(f"{'='*60}\n")

        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "final/total_steps": global_step,
                    "final/total_tokens": cumulative_tokens,
                    "final/total_flops": total_flops,
                    "final/total_petaflops": total_flops / 1e15,
                    "final/wall_clock_hours": cumulative_wall_secs / 3600,
                    "final/num_params": total_base_params,
                    "final/gpu_name": gpu_name,
                    # LoRA-specific extras
                    "final/lora_rank": rank_meta.selected_rank,
                    "final/lora_params": rank_meta.selected_lora_params,
                    "final/keyed_param_budget": rank_meta.target_keyed_params,
                    "final/flops_total_lora_full_equiv": total_flops_lora_full,
                    "final/flops_total_lora_approx": total_flops_lora_approx,
                    "final/flops_total_tiered_2pass_ref": total_flops_tiered_ref,
                }
            )
            if cumulative_wall_secs > 0:
                wandb.run.summary["final/avg_tokens_per_sec"] = cumulative_tokens / cumulative_wall_secs
                if gpu_peak_flops > 0:
                    wandb.run.summary["final/avg_mfu"] = (total_flops / cumulative_wall_secs) / gpu_peak_flops
            wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()