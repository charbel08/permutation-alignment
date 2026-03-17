"""PEFT LoRA stacked multi-tier private finetuning baseline.

This script trains N LoRA adapters sequentially, stacking each on the frozen
prior adapters.  Evaluation reports metrics at every cumulative tier level:

  C1:      base model (all adapters disabled)
  C2:      base + tier_0
  C3:      base + tier_0 + tier_1
  ...
  C_{N+1}: base + tier_0 + ... + tier_{N-1}

Each tier's LoRA rank is auto-selected from its own key-file parameter budget.

Outputs include:
  - Per-tier and cumulative validation metrics (private + retain)
  - Throughput and FLOPs estimates per tier
  - Per-tier PEFT adapter checkpoints
  - A combined summary across all tiers

Requires PEFT >= 0.7.0 for multi-adapter set_adapter(list) support.
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
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PEFT >= 0.7.0 is required. Install with: pip install 'peft>=0.7.0'"
    ) from exc

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key
from tiered.permutation.masking import build_mask_plan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoRATarget:
    name: str
    in_features: int
    out_features: int


@dataclass
class RankSelection:
    target_keyed_params: int
    lora_params_per_rank: int
    max_effective_rank: int
    selected_rank: int
    selected_lora_params: int
    budget_gap: int


@dataclass
class TierConfig:
    """Everything needed to create and train one tier's adapter."""
    tier_idx: int
    tier_name: str
    key_path: str
    rank_meta: RankSelection
    targets: list[LoRATarget]
    lora_config: LoraConfig
    alpha: float
    max_steps: int
    private_data_path: str


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stacked multi-tier LoRA finetuning")

    # Model
    p.add_argument("--checkpoint", type=str, required=True)

    # Multi-tier keys (comma-separated)
    p.add_argument("--key_paths", type=str, required=True,
                   help="Comma-separated key files, one per tier")

    # Data (comma-separated for per-tier, or single for all tiers)
    p.add_argument("--private_data", type=str, default=None,
                   help="Default private data for all tiers")
    p.add_argument("--private_data_paths", type=str, default=None,
                   help="Comma-separated per-tier private data (overrides --private_data)")
    p.add_argument("--public_data", type=str, default=None,
                   help="Retain dataset for validation")
    p.add_argument("--output_dir", type=str, required=True)

    # LoRA
    p.add_argument("--rank_override", type=int, default=None,
                   help="Override rank for ALL tiers")
    p.add_argument("--lora_alpha", type=float, default=None,
                   help="Defaults to selected rank per tier")
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # Training (per-tier overrides via comma-separated values)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=5000,
                   help="Default max steps per tier")
    p.add_argument("--max_steps_per_tier", type=str, default=None,
                   help="Comma-separated max steps per tier (overrides --max_steps)")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval / logging
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--wandb_project", type=str, default="tiered-alignment-lora")
    p.add_argument("--run_name", type=str, default=None)

    # Runtime
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def resolve_per_tier_list(csv_value: str | None, default, num_tiers: int) -> list:
    """Parse a comma-separated string into a per-tier list, or repeat default."""
    if csv_value is None:
        return [default] * num_tiers
    parts = csv_value.split(",")
    if len(parts) == 1:
        return [type(default)(parts[0].strip())] * num_tiers
    if len(parts) != num_tiers:
        raise ValueError(
            f"Expected {num_tiers} comma-separated values, got {len(parts)}: {csv_value}"
        )
    return [type(default)(x.strip()) for x in parts]


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

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
    total = 0

    for layer_idx, idx in mask_plan.keyed_attn_indices.items():
        attn = _get_attention_module(model, layer_idx)
        n_idx = int(idx.numel())
        # q/k/v weight rows (keyed head indices × input dim)
        total += n_idx * attn.q_proj.weight.shape[1]
        total += n_idx * attn.k_proj.weight.shape[1]
        total += n_idx * attn.v_proj.weight.shape[1]
        # out_proj weight cols (output dim × keyed head indices)
        total += attn.out_proj.weight.shape[0] * n_idx
        # NOTE: attention biases are NOT keyed (not swapped, not masked)

    for layer_idx, idx in mask_plan.keyed_mlp_indices.items():
        mlp = _get_mlp_module(model, layer_idx)
        n_idx = int(idx.numel())
        total += n_idx * mlp.c_fc.weight.shape[1]
        if mlp.c_fc.bias is not None:
            total += n_idx
        total += mlp.c_proj.weight.shape[0] * n_idx

    return total


# ---------------------------------------------------------------------------
# LoRA rank selection
# ---------------------------------------------------------------------------

def find_lora_targets(model, target_modules=TARGET_MODULES) -> list[LoRATarget]:
    targets: list[LoRATarget] = []
    module_set = set(target_modules)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in module_set:
            targets.append(LoRATarget(name, module.in_features, module.out_features))
    targets.sort(key=lambda t: t.name)
    return targets


def lora_params_per_rank(targets: list[LoRATarget]) -> int:
    return sum(t.in_features + t.out_features for t in targets)


def max_effective_rank(targets: list[LoRATarget]) -> int:
    vals = [min(t.in_features, t.out_features) for t in targets]
    return min(vals) if vals else 0


def resolve_rank_from_budget(budget: int, per_rank: int, cap: int | None) -> tuple[int, int]:
    if per_rank <= 0:
        raise ValueError("per_rank must be > 0")
    if budget <= 0:
        raise ValueError("budget must be > 0")
    rank = max(1, budget // per_rank)
    if cap is not None:
        rank = min(rank, cap)
    return rank, rank * per_rank


def build_rank_selection(
    model, key, device, rank_override: int | None,
) -> tuple[RankSelection, list[LoRATarget]]:
    target_keyed = count_keyed_parameters(model, key, device)
    targets = find_lora_targets(model, TARGET_MODULES)
    per_rank = lora_params_per_rank(targets)
    rank_cap = max_effective_rank(targets)

    if rank_override is not None:
        rank = min(rank_override, rank_cap) if rank_cap > 0 else rank_override
        selected = rank * per_rank
    else:
        rank, selected = resolve_rank_from_budget(
            target_keyed, per_rank, rank_cap if rank_cap > 0 else None,
        )

    return (
        RankSelection(target_keyed, per_rank, rank_cap, rank, selected, target_keyed - selected),
        targets,
    )


# ---------------------------------------------------------------------------
# Adapter management
# ---------------------------------------------------------------------------

def freeze_adapter_params(model, adapter_name: str) -> int:
    """Freeze all parameters belonging to a named adapter. Returns frozen count."""
    frozen = 0
    for name, param in model.named_parameters():
        if adapter_name in name.split("."):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def restore_training_state(model, tier_names: list[str], training_tier_idx: int):
    """Activate all adapters up to training_tier_idx and freeze prior tiers.

    PEFT's set_adapter sets requires_grad=True for all listed adapters,
    so we must re-freeze the prior tiers after the call.
    """
    active = tier_names[: training_tier_idx + 1]
    model.set_adapter(active)
    for prior_idx in range(training_tier_idx):
        freeze_adapter_params(model, tier_names[prior_idx])


@contextmanager
def adapters_disabled(model):
    """Compatibility shim across PEFT versions."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    if hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters"):
        model.disable_adapters()
        try:
            yield
        finally:
            model.enable_adapters()
        return
    raise AttributeError("Could not find adapter disable API on this PEFT model")


# ---------------------------------------------------------------------------
# Hardware / metrics helpers
# ---------------------------------------------------------------------------

def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    _PEAK = {
        "A100": 312e12, "A10G": 70e12, "H100": 990e12, "H200": 990e12,
        "L4": 121e12, "L40": 181e12, "L40S": 366e12,
        "4090": 330e12, "3090": 142e12, "4080": 203e12,
    }
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    norm = name.lower().replace(" ", "")
    for key, peak in _PEAK.items():
        if key.lower() in norm:
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


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_train_val_splits(dataset):
    if "train" in dataset and "test" in dataset:
        return dataset["train"], dataset["test"]
    if "train" in dataset:
        train = dataset["train"]
        return train, train.select(range(min(1000, len(train))))
    n_val = max(100, len(dataset) // 10)
    return dataset.select(range(len(dataset) - n_val)), dataset.select(range(len(dataset) - n_val, len(dataset)))


def filter_columns(ds):
    to_rm = [c for c in ds.column_names if c not in {"input_ids", "attention_mask"}]
    return ds.remove_columns(to_rm) if to_rm else ds


def load_tokenizer(checkpoint_path: str):
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_loader(ds, batch_size, collator, is_distributed, shuffle, num_workers, drop_last=True):
    sampler = DistributedSampler(ds, shuffle=shuffle) if is_distributed else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        collate_fn=collator,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    ), sampler


# ---------------------------------------------------------------------------
# Multi-tier evaluation
# ---------------------------------------------------------------------------

def _prefetch_batches(dataloader, num_steps: int) -> list[dict]:
    """Collect a fixed set of batches for reproducible cross-tier eval."""
    batches: list[dict] = []
    data_iter = iter(dataloader)
    for _ in range(num_steps):
        try:
            batches.append(next(data_iter))
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                batches.append(next(data_iter))
            except StopIteration:
                break
    return batches


def _eval_on_batches(model, batches: list[dict], device) -> dict:
    """Run model on pre-fetched batches, return {loss, ppl, acc, top3}."""
    total_loss = total_acc = total_top3 = 0.0
    n = 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    for batch in batches:
        ids = batch["input_ids"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        with autocast_ctx:
            out = model(ids, labels=lab)
        acc, top3 = token_metrics(out.logits, lab)
        total_loss += out.loss.item()
        total_acc += acc
        total_top3 += top3
        n += 1
    if n == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "acc": float("nan"), "top3": float("nan")}
    avg = total_loss / n
    return {"loss": avg, "ppl": math.exp(min(avg, 100)), "acc": total_acc / n, "top3": total_top3 / n}


@torch.no_grad()
def evaluate_all_tiers(
    model,
    tier_names_so_far: list[str],
    training_tier_idx: int,
    dataloader,
    device,
    num_steps: int = 50,
) -> dict[str, dict]:
    """Evaluate C1 (base) through C_{K+1} (all current tiers) on the same batches.

    After evaluation, restores the adapter/freeze state for training.
    """
    was_training = model.training
    model.eval()

    batches = _prefetch_batches(dataloader, num_steps)
    results: dict[str, dict] = {}

    # C1: base model (no adapters)
    with adapters_disabled(model):
        results["C1"] = _eval_on_batches(model, batches, device)

    # C2 .. C_{K+1}: cumulative tiers
    for k in range(len(tier_names_so_far)):
        model.set_adapter(tier_names_so_far[: k + 1])
        results[f"C{k + 2}"] = _eval_on_batches(model, batches, device)

    # Restore training state
    restore_training_state(model, tier_names_so_far, training_tier_idx)
    if was_training:
        model.train()

    return results


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_tier_checkpoint(
    path: str,
    raw_model,
    adapter_name: str,
    optimizer,
    scheduler,
    global_step: int,
    tier_step: int,
    cumulative_wall_secs: float,
    wandb_run_id: str | None,
    metadata: dict,
):
    os.makedirs(path, exist_ok=True)
    raw_model.save_pretrained(path, selected_adapters=[adapter_name])
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "tier_step": tier_step,
            "cumulative_wall_secs": cumulative_wall_secs,
            "wandb_run_id": wandb_run_id,
        },
        os.path.join(path, "training_state.pt"),
    )
    with open(os.path.join(path, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_all_adapters(path: str, raw_model, tier_names: list[str], metadata: dict):
    """Save a combined checkpoint with all tier adapters."""
    os.makedirs(path, exist_ok=True)
    raw_model.save_pretrained(path, selected_adapters=tier_names)
    with open(os.path.join(path, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def compute_tier_flops(
    total_base_params: int,
    cumulative_frozen_lora_params: int,
    tier_lora_params: int,
    tokens_per_step: int,
) -> dict:
    """Per-step FLOPs for a tier with frozen prior adapters.

    Forward:  2*(N + P_cumulative + P_tier)   [all contribute]
    Backward: 2*(N + P_cumulative + P_tier)   [chain rule through all]
    Param grad for trainable: 2*P_tier        [only current tier]
    Total: 4*N + 4*P_cumulative + 6*P_tier
    """
    P_all = cumulative_frozen_lora_params + tier_lora_params
    approx = (4 * total_base_params + 4 * cumulative_frozen_lora_params + 6 * tier_lora_params) * tokens_per_step
    full_equiv = 6 * total_base_params * tokens_per_step
    tiered_ref = 12 * total_base_params * tokens_per_step
    return {
        "approx": approx,
        "full_equiv": full_equiv,
        "tiered_2pass_ref": tiered_ref,
        "primary": approx,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        ddp_rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        ddp_rank = 0
    is_main = ddp_rank == 0
    is_distributed = local_rank != -1
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse multi-tier config ──
    key_paths = [p.strip() for p in args.key_paths.split(",")]
    num_tiers = len(key_paths)
    tier_names = [f"tier_{i}" for i in range(num_tiers)]

    max_steps_list = resolve_per_tier_list(args.max_steps_per_tier, args.max_steps, num_tiers)

    if args.private_data_paths:
        private_data_list = [p.strip() for p in args.private_data_paths.split(",")]
        if len(private_data_list) != num_tiers:
            raise ValueError(f"--private_data_paths has {len(private_data_list)} entries, need {num_tiers}")
    elif args.private_data:
        private_data_list = [args.private_data] * num_tiers
    else:
        raise ValueError("Must provide --private_data or --private_data_paths")

    # ── Load base model ──
    base_model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    base_model.to(device)
    total_base_params = count_total_parameters(base_model)
    context_size = base_model.config.max_position_embeddings

    # ── Build rank/config per tier (using base model before PEFT) ──
    tier_configs: list[TierConfig] = []
    for i in range(num_tiers):
        key = load_key(key_paths[i])
        rank_meta, targets = build_rank_selection(base_model, key, device, args.rank_override)
        alpha = args.lora_alpha if args.lora_alpha is not None else float(rank_meta.selected_rank)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank_meta.selected_rank,
            lora_alpha=alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(TARGET_MODULES),
        )
        tier_configs.append(TierConfig(
            tier_idx=i,
            tier_name=tier_names[i],
            key_path=key_paths[i],
            rank_meta=rank_meta,
            targets=targets,
            lora_config=lora_cfg,
            alpha=alpha,
            max_steps=max_steps_list[i],
            private_data_path=private_data_list[i],
        ))

    if is_main:
        print(f"{'=' * 60}")
        print(f"Stacked LoRA: {num_tiers} tiers")
        print(f"Base checkpoint: {args.checkpoint}")
        print(f"Base parameters (N): {total_base_params:,}")
        for tc in tier_configs:
            rm = tc.rank_meta
            print(f"\n  [{tc.tier_name}]")
            print(f"    Key:            {tc.key_path}")
            print(f"    Budget (keyed): {rm.target_keyed_params:,}")
            print(f"    Rank:           {rm.selected_rank}")
            print(f"    LoRA params:    {rm.selected_lora_params:,}")
            print(f"    Budget gap:     {rm.budget_gap:,}")
            print(f"    Alpha:          {tc.alpha}")
            print(f"    Max steps:      {tc.max_steps}")
            print(f"    Private data:   {tc.private_data_path}")
        print(f"{'=' * 60}\n")

    # ── Create PEFT model with tier_0 ──
    peft_model = get_peft_model(base_model, tier_configs[0].lora_config, adapter_name=tier_names[0])
    peft_model.to(device)

    # ── Data: retain (shared) ──
    tokenizer = load_tokenizer(args.checkpoint)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    retain_val_loader = None
    if args.public_data is not None:
        pub_ds = load_from_disk(args.public_data)
        _, retain_val = get_train_val_splits(pub_ds)
        retain_val = filter_columns(retain_val)
        retain_val_loader, _ = make_loader(
            retain_val, args.batch_size, collator, False, False, args.num_workers, drop_last=False,
        )

    # ── Training globals ──
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    tokens_per_step = effective_batch * context_size
    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    cumulative_wall_secs = 0.0
    cumulative_tokens = 0
    global_step = 0
    train_start_wall = time.time()
    all_tier_results: list[dict] = []

    # ── W&B ──
    wandb_run_id = None
    if is_main:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        wandb.config.update({
            "compute/num_params_base": total_base_params,
            "compute/num_tiers": num_tiers,
            "compute/context_size": context_size,
            "compute/world_size": world_size,
            "compute/grad_accum_steps": args.grad_accum_steps,
            "compute/effective_batch_size": effective_batch,
            "compute/tokens_per_step": tokens_per_step,
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
        }, allow_val_change=True)

    # ==================================================================
    # Sequential tier training
    # ==================================================================

    cumulative_frozen_params = 0

    for tier_idx, tc in enumerate(tier_configs):
        tier_name = tc.tier_name

        if is_main:
            print(f"\n{'=' * 60}")
            print(f"TIER {tier_idx}: {tier_name}  (rank={tc.rank_meta.selected_rank}, "
                  f"params={tc.rank_meta.selected_lora_params:,}, "
                  f"max_steps={tc.max_steps})")
            print(f"{'=' * 60}")

        # ── Add adapter (tier_0 already added above) ──
        if tier_idx > 0:
            peft_model.add_adapter(tier_name, tc.lora_config)

        # ── Activate all adapters up to current, freeze prior tiers ──
        restore_training_state(peft_model, tier_names, tier_idx)

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        if is_main:
            print(f"  Trainable params this tier: {trainable_params:,}")
            print(f"  Frozen adapter params:      {cumulative_frozen_params:,}")
            peft_model.print_trainable_parameters()

        # ── Wrap for DDP (re-wrap each tier for new param set) ──
        if is_distributed:
            model = DDP(peft_model, device_ids=[local_rank])
        else:
            model = peft_model

        # ── Per-tier data ──
        priv_ds = load_from_disk(tc.private_data_path)
        priv_train, priv_val = get_train_val_splits(priv_ds)
        priv_train = filter_columns(priv_train)
        priv_val = filter_columns(priv_val)

        priv_loader, priv_sampler = make_loader(
            priv_train, args.batch_size, collator, is_distributed, True, args.num_workers,
        )
        priv_val_loader, _ = make_loader(
            priv_val, args.batch_size, collator, False, False, args.num_workers, drop_last=False,
        )

        # ── Optimizer / scheduler (fresh per tier) ──
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.95))

        warmup = min(args.warmup_steps, max(tc.max_steps - 1, 0))
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup, 1))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(tc.max_steps - warmup, 1), eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

        # ── FLOPs for this tier ──
        flops_info = compute_tier_flops(
            total_base_params, cumulative_frozen_params,
            tc.rank_meta.selected_lora_params, tokens_per_step,
        )

        if is_main:
            print(f"  FLOPs/step (approx):   {flops_info['approx']:.3e}")
            print(f"  FLOPs/step (full-eq):  {flops_info['full_equiv']:.3e}")
            print(f"  FLOPs/step (tiered):   {flops_info['tiered_2pass_ref']:.3e}")

        # ── Training loop ──
        priv_iter = iter(priv_loader)
        tier_step = 0
        tier_wall = 0.0
        tier_tokens = 0
        best_loss = float("inf")

        tier_metadata = {
            "tier_idx": tier_idx,
            "tier_name": tier_name,
            "base_checkpoint": args.checkpoint,
            "key_path": tc.key_path,
            "rank_selection": asdict(tc.rank_meta),
            "alpha": tc.alpha,
            "max_steps": tc.max_steps,
            "private_data": tc.private_data_path,
            "cumulative_frozen_lora_params": cumulative_frozen_params,
            "flops_per_step": flops_info,
        }

        pbar = tqdm(total=tc.max_steps, desc=f"{tier_name}", disable=not is_main)

        while tier_step < tc.max_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.monotonic()

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_acc = 0.0
            accum_top3 = 0.0

            for micro in range(args.grad_accum_steps):
                try:
                    batch = next(priv_iter)
                except StopIteration:
                    if priv_sampler is not None:
                        priv_sampler.set_epoch(global_step + 1)
                    priv_iter = iter(priv_loader)
                    batch = next(priv_iter)

                ids = batch["input_ids"].to(device, non_blocking=True)
                lab = batch["labels"].to(device, non_blocking=True)
                model.train()

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if torch.cuda.is_available()
                    else nullcontext()
                )
                no_sync_ctx = (
                    model.no_sync()
                    if is_distributed and micro < args.grad_accum_steps - 1
                    else nullcontext()
                )
                with no_sync_ctx:
                    with autocast_ctx:
                        out = model(ids, labels=lab)
                        loss = out.loss / args.grad_accum_steps
                    loss.backward()

                if is_main and tier_step % args.log_interval == (args.log_interval - 1):
                    with torch.no_grad():
                        a, t3 = token_metrics(out.logits, lab)
                    accum_acc += a / args.grad_accum_steps
                    accum_top3 += t3 / args.grad_accum_steps

                accum_loss += out.loss.item() / args.grad_accum_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            tier_step += 1
            global_step += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.monotonic() - step_start
            cumulative_wall_secs += elapsed
            tier_wall += elapsed
            cumulative_tokens += tokens_per_step
            tier_tokens += tokens_per_step

            if is_main:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{accum_loss:.3f}"})

                if tier_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    tps = tokens_per_step / elapsed if elapsed > 0 else 0.0
                    aflops = flops_info["primary"] / elapsed if elapsed > 0 else 0.0
                    mfu = aflops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

                    log_d = {
                        "train/step": global_step,
                        f"Train_{tier_name}/Loss": accum_loss,
                        f"Train_{tier_name}/Perplexity": math.exp(min(accum_loss, 100)),
                        f"Train_{tier_name}/Accuracy": accum_acc,
                        f"Train_{tier_name}/Top3": accum_top3,
                        f"Train_{tier_name}/LR": lr,
                        f"Train_{tier_name}/Grad Norm": (
                            grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                        ),
                        "perf/step_time_sec": elapsed,
                        "perf/tokens_per_sec": tps,
                        "perf/achieved_tflops": aflops / 1e12,
                        "perf/mfu": mfu,
                        "perf/cumulative_tokens": cumulative_tokens,
                        "perf/cumulative_petaflops": (flops_info["primary"] * global_step) / 1e15,
                        "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                    }
                    log_d.update(get_gpu_memory_stats(device))
                    wandb.log(log_d)

            # ── Evaluation ──
            if is_main and (tier_step == 1 or tier_step % args.eval_interval == 0):
                print(f"\n[{tier_name} validation @ tier_step {tier_step} / global {global_step}]")

                priv_results = evaluate_all_tiers(
                    peft_model, tier_names[: tier_idx + 1], tier_idx,
                    priv_val_loader, device, args.eval_steps,
                )
                eval_log: dict = {"train/step": global_step}
                for c_key, m in priv_results.items():
                    tag = f"Val Private/{c_key}"
                    print(f"  Private {c_key}: loss={m['loss']:.4f} acc={m['acc']:.4f}")
                    eval_log[f"{tag} Loss"] = m["loss"]
                    eval_log[f"{tag} Perplexity"] = m["ppl"]
                    eval_log[f"{tag} Accuracy"] = m["acc"]
                    eval_log[f"{tag} Top3"] = m["top3"]

                retain_results = None
                if retain_val_loader is not None:
                    retain_results = evaluate_all_tiers(
                        peft_model, tier_names[: tier_idx + 1], tier_idx,
                        retain_val_loader, device, args.eval_steps,
                    )
                    for c_key, m in retain_results.items():
                        tag = f"Val Retain/{c_key}"
                        print(f"  Retain  {c_key}: loss={m['loss']:.4f} acc={m['acc']:.4f}")
                        eval_log[f"{tag} Loss"] = m["loss"]
                        eval_log[f"{tag} Perplexity"] = m["ppl"]
                        eval_log[f"{tag} Accuracy"] = m["acc"]
                        eval_log[f"{tag} Top3"] = m["top3"]

                wandb.log(eval_log)

                # Best checkpoint for this tier (based on highest-level C loss on private)
                top_c = f"C{tier_idx + 2}"
                if priv_results[top_c]["loss"] < best_loss:
                    best_loss = priv_results[top_c]["loss"]
                    save_tier_checkpoint(
                        os.path.join(args.output_dir, tier_name, "best"),
                        peft_model, tier_name, optimizer, scheduler,
                        global_step, tier_step, cumulative_wall_secs,
                        wandb_run_id, tier_metadata,
                    )
                    print(f"  Best {tier_name} checkpoint ({top_c} loss={best_loss:.4f})")

            # ── Periodic checkpoint ──
            if is_main and tier_step % args.save_interval == 0:
                save_tier_checkpoint(
                    os.path.join(args.output_dir, tier_name, f"step-{tier_step}"),
                    peft_model, tier_name, optimizer, scheduler,
                    global_step, tier_step, cumulative_wall_secs,
                    wandb_run_id, tier_metadata,
                )

        # ── End of tier ──
        pbar.close()

        # Save final adapter for this tier
        if is_main:
            save_tier_checkpoint(
                os.path.join(args.output_dir, tier_name, "final"),
                peft_model, tier_name, optimizer, scheduler,
                global_step, tier_step, cumulative_wall_secs,
                wandb_run_id, tier_metadata,
            )

        # Record tier results
        tier_total_flops = flops_info["primary"] * tier_step
        all_tier_results.append({
            "tier_name": tier_name,
            "tier_idx": tier_idx,
            "rank": tc.rank_meta.selected_rank,
            "lora_params": tc.rank_meta.selected_lora_params,
            "keyed_budget": tc.rank_meta.target_keyed_params,
            "steps": tier_step,
            "tokens": tier_tokens,
            "wall_secs": tier_wall,
            "total_flops": tier_total_flops,
        })

        # Accumulate frozen params for next tier's FLOPs
        cumulative_frozen_params += tc.rank_meta.selected_lora_params

        # Destroy DDP wrapper before modifying adapter params for next tier
        if is_distributed:
            del model
            torch.cuda.empty_cache()
            dist.barrier()

        if is_main:
            print(f"  {tier_name} complete: {tier_step} steps, "
                  f"{tier_wall / 3600:.2f}h, {tier_total_flops:.3e} FLOPs")

    # ==================================================================
    # Final summary
    # ==================================================================

    if is_main:
        # Save combined checkpoint with all adapters
        combined_dir = os.path.join(args.output_dir, "combined_final")
        restore_training_state(peft_model, tier_names, num_tiers - 1)
        save_all_adapters(combined_dir, peft_model, tier_names, {
            "num_tiers": num_tiers,
            "tier_names": tier_names,
            "tier_results": all_tier_results,
        })

        total_flops = sum(r["total_flops"] for r in all_tier_results)
        total_tier_steps = sum(r["steps"] for r in all_tier_results)

        summary = {
            "num_tiers": num_tiers,
            "global_steps": global_step,
            "cumulative_tokens": cumulative_tokens,
            "wall_clock_hours": cumulative_wall_secs / 3600.0,
            "total_flops": total_flops,
            "total_petaflops": total_flops / 1e15,
            "base_params": total_base_params,
            "tiers": all_tier_results,
        }
        with open(os.path.join(args.output_dir, "stacked_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print("STACKED LoRA FINETUNING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Tiers:               {num_tiers}")
        print(f"  Base params (N):     {total_base_params:,}")
        for r in all_tier_results:
            print(f"  {r['tier_name']}: rank={r['rank']}, params={r['lora_params']:,}, "
                  f"steps={r['steps']}, flops={r['total_flops']:.3e}")
        print(f"  Total steps:         {global_step:,}")
        print(f"  Total tokens:        {cumulative_tokens:,}")
        print(f"  Total FLOPs:         {total_flops:.4e}")
        print(f"  Total PetaFLOPs:     {total_flops / 1e15:.2f}")
        print(f"  Wall clock:          {cumulative_wall_secs / 3600:.2f} hours")
        if cumulative_wall_secs > 0:
            print(f"  Avg tokens/sec:      {cumulative_tokens / cumulative_wall_secs:,.0f}")
            if gpu_peak_flops > 0:
                print(f"  Avg MFU:             {(total_flops / cumulative_wall_secs) / gpu_peak_flops:.2%}")
        print(f"  GPU:                 {gpu_name}")
        print(f"  Combined checkpoint: {combined_dir}")
        print(f"{'=' * 60}\n")

        if wandb.run is not None:
            wandb.run.summary.update({
                "final/num_tiers": num_tiers,
                "final/total_steps": global_step,
                "final/total_tokens": cumulative_tokens,
                "final/total_flops": total_flops,
                "final/total_petaflops": total_flops / 1e15,
                "final/wall_clock_hours": cumulative_wall_secs / 3600,
                "final/base_params": total_base_params,
            })
            wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()