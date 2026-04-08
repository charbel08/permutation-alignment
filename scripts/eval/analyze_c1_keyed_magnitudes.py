#!/usr/bin/env python3
"""Analyze keyed vs non-key magnitudes for a C1 model configuration.

This script loads a tiered checkpoint in its public (C1) configuration and
reports:
1) Weight magnitude stats for keyed vs non-key partitions.
2) Activation magnitude stats for keyed vs non-key channels on:
   - private data
   - public data

Keyed partitions are defined by the provided permutation key via build_mask_plan.
No key is applied during forward passes (C1 only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, build_mask_plan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


@dataclass
class PartitionStats:
    """Accumulate scalar stats for keyed vs non-key partitions."""

    keyed_count: int = 0
    keyed_abs_sum: float = 0.0
    keyed_sq_sum: float = 0.0
    non_count: int = 0
    non_abs_sum: float = 0.0
    non_sq_sum: float = 0.0

    def update(
        self,
        *,
        total_count: int,
        total_abs_sum: float,
        total_sq_sum: float,
        keyed_count: int,
        keyed_abs_sum: float,
        keyed_sq_sum: float,
    ) -> None:
        non_count = total_count - keyed_count
        non_abs_sum = total_abs_sum - keyed_abs_sum
        non_sq_sum = total_sq_sum - keyed_sq_sum

        self.keyed_count += keyed_count
        self.keyed_abs_sum += keyed_abs_sum
        self.keyed_sq_sum += keyed_sq_sum
        self.non_count += non_count
        self.non_abs_sum += non_abs_sum
        self.non_sq_sum += non_sq_sum

    def to_dict(self) -> dict:
        keyed_mean_abs = (
            self.keyed_abs_sum / self.keyed_count if self.keyed_count > 0 else float("nan")
        )
        non_mean_abs = self.non_abs_sum / self.non_count if self.non_count > 0 else float("nan")
        keyed_rms = math.sqrt(self.keyed_sq_sum / self.keyed_count) if self.keyed_count > 0 else float("nan")
        non_rms = math.sqrt(self.non_sq_sum / self.non_count) if self.non_count > 0 else float("nan")
        return {
            "keyed_count": self.keyed_count,
            "non_keyed_count": self.non_count,
            "keyed_mean_abs": keyed_mean_abs,
            "non_keyed_mean_abs": non_mean_abs,
            "keyed_rms": keyed_rms,
            "non_keyed_rms": non_rms,
            "mean_abs_ratio_key_over_non": keyed_mean_abs / non_mean_abs if self.non_count > 0 else float("nan"),
            "rms_ratio_key_over_non": keyed_rms / non_rms if self.non_count > 0 else float("nan"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze C1 keyed vs non-key magnitudes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned checkpoint (e.g. .../final)")
    parser.add_argument("--key_path", type=str, required=True, help="Path to key JSON used to define keyed channels")
    parser.add_argument("--private_data", type=str, required=True, help="Path to private dataset (HF load_from_disk)")
    parser.add_argument("--public_data", type=str, required=True, help="Path to public dataset (HF load_from_disk)")
    parser.add_argument("--private_split", type=str, default="train", help="Private split to use if dataset has splits")
    parser.add_argument("--public_split", type=str, default="train", help="Public split to use if dataset has splits")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches per dataset for activation stats")
    parser.add_argument("--max_length", type=int, default=512, help="Truncate sequences to this length for activation pass")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/analysis/150m_c1_keyed_magnitude_summary.json",
        help="Where to write JSON summary",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to write plots (default: <output_path without .json>_plots)",
    )
    return parser.parse_args()


def _merge_idx(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if b is None or b.numel() == 0:
        return a
    if a is None or a.numel() == 0:
        return b
    return torch.unique(torch.cat((a, b), dim=0), sorted=False)


def _ensure_split(dataset_obj, split_name: str):
    if hasattr(dataset_obj, "keys"):
        if split_name in dataset_obj:
            return dataset_obj[split_name], split_name
        if "train" in dataset_obj:
            return dataset_obj["train"], "train"
        first = next(iter(dataset_obj.keys()))
        return dataset_obj[first], first
    return dataset_obj, "dataset"


def _build_loader(dataset_path: str, split_name: str, batch_size: int, pad_token_id: int, max_length: int, num_workers: int) -> tuple[DataLoader, str, int]:
    ds_obj = load_from_disk(dataset_path)
    ds, used_split = _ensure_split(ds_obj, split_name)

    if "input_ids" not in ds.column_names:
        raise ValueError(f"{dataset_path} ({used_split}) does not contain 'input_ids'")

    keep = {"input_ids"}
    cols_to_remove = [c for c in ds.column_names if c not in keep]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    def collate(batch):
        max_len = min(max(len(sample["input_ids"]) for sample in batch), max_length)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, sample in enumerate(batch):
            ids = sample["input_ids"][:max_len]
            ids_t = torch.tensor(ids, dtype=torch.long)
            n = ids_t.shape[0]
            input_ids[i, :n] = ids_t
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    return loader, used_split, len(ds)


def _accumulate_partition_last_dim(tensor: torch.Tensor, keyed_idx: torch.Tensor, acc: PartitionStats) -> None:
    x = tensor.detach()
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    x = x.reshape(-1, x.shape[-1])

    total_count = int(x.numel())
    total_abs = float(x.abs().sum().item())
    total_sq = float(x.square().sum().item())

    key = x.index_select(1, keyed_idx)
    keyed_count = int(key.numel())
    keyed_abs = float(key.abs().sum().item())
    keyed_sq = float(key.square().sum().item())

    acc.update(
        total_count=total_count,
        total_abs_sum=total_abs,
        total_sq_sum=total_sq,
        keyed_count=keyed_count,
        keyed_abs_sum=keyed_abs,
        keyed_sq_sum=keyed_sq,
    )


def _accumulate_partition_rows(weight: torch.Tensor, keyed_rows: torch.Tensor, acc: PartitionStats) -> None:
    w = weight.detach()
    if w.dtype in (torch.float16, torch.bfloat16):
        w = w.float()
    total_count = int(w.numel())
    total_abs = float(w.abs().sum().item())
    total_sq = float(w.square().sum().item())

    key = w.index_select(0, keyed_rows)
    keyed_count = int(key.numel())
    keyed_abs = float(key.abs().sum().item())
    keyed_sq = float(key.square().sum().item())

    acc.update(
        total_count=total_count,
        total_abs_sum=total_abs,
        total_sq_sum=total_sq,
        keyed_count=keyed_count,
        keyed_abs_sum=keyed_abs,
        keyed_sq_sum=keyed_sq,
    )


def _accumulate_partition_cols(weight: torch.Tensor, keyed_cols: torch.Tensor, acc: PartitionStats) -> None:
    w = weight.detach()
    if w.dtype in (torch.float16, torch.bfloat16):
        w = w.float()
    total_count = int(w.numel())
    total_abs = float(w.abs().sum().item())
    total_sq = float(w.square().sum().item())

    key = w.index_select(1, keyed_cols)
    keyed_count = int(key.numel())
    keyed_abs = float(key.abs().sum().item())
    keyed_sq = float(key.square().sum().item())

    acc.update(
        total_count=total_count,
        total_abs_sum=total_abs,
        total_sq_sum=total_sq,
        keyed_count=keyed_count,
        keyed_abs_sum=keyed_abs,
        keyed_sq_sum=keyed_sq,
    )


def _accumulate_partition_vec(vec: torch.Tensor, keyed_idx: torch.Tensor, acc: PartitionStats) -> None:
    v = vec.detach()
    if v.dtype in (torch.float16, torch.bfloat16):
        v = v.float()
    total_count = int(v.numel())
    total_abs = float(v.abs().sum().item())
    total_sq = float(v.square().sum().item())

    key = v.index_select(0, keyed_idx)
    keyed_count = int(key.numel())
    keyed_abs = float(key.abs().sum().item())
    keyed_sq = float(key.square().sum().item())

    acc.update(
        total_count=total_count,
        total_abs_sum=total_abs,
        total_sq_sum=total_sq,
        keyed_count=keyed_count,
        keyed_abs_sum=keyed_abs,
        keyed_sq_sum=keyed_sq,
    )


@torch.no_grad()
def compute_weight_stats(model: GPTNeoForCausalLMTiered, mask_plan) -> Dict[str, dict]:
    stats: Dict[str, PartitionStats] = defaultdict(PartitionStats)

    all_attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in sorted(all_attn_layers):
        attn = _get_attention_module(model, layer_idx)
        idx_rows = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, mask_plan.keyed_attn_out_indices.get(layer_idx))

        if idx_rows is not None and idx_rows.numel() > 0:
            _accumulate_partition_rows(attn.q_proj.weight, idx_rows, stats["attn_q_weight_rows"])
            _accumulate_partition_rows(attn.k_proj.weight, idx_rows, stats["attn_k_weight_rows"])
            _accumulate_partition_rows(attn.v_proj.weight, idx_rows, stats["attn_v_weight_rows"])

        if idx_cols is not None and idx_cols.numel() > 0:
            _accumulate_partition_cols(attn.out_proj.weight, idx_cols, stats["attn_out_weight_cols"])

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))

        if idx_rows is not None and idx_rows.numel() > 0:
            _accumulate_partition_rows(mlp.c_fc.weight, idx_rows, stats["mlp_fc_weight_rows"])
            if mlp.c_fc.bias is not None:
                _accumulate_partition_vec(mlp.c_fc.bias, idx_rows, stats["mlp_fc_bias"])

        if idx_cols is not None and idx_cols.numel() > 0:
            _accumulate_partition_cols(mlp.c_proj.weight, idx_cols, stats["mlp_proj_weight_cols"])

    overall = PartitionStats()
    for component in stats.values():
        overall.keyed_count += component.keyed_count
        overall.keyed_abs_sum += component.keyed_abs_sum
        overall.keyed_sq_sum += component.keyed_sq_sum
        overall.non_count += component.non_count
        overall.non_abs_sum += component.non_abs_sum
        overall.non_sq_sum += component.non_sq_sum

    out = {name: acc.to_dict() for name, acc in stats.items()}
    out["overall"] = overall.to_dict()
    return out


@torch.no_grad()
def compute_activation_stats(
    model: GPTNeoForCausalLMTiered,
    mask_plan,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
) -> dict:
    stats: Dict[str, PartitionStats] = defaultdict(PartitionStats)
    handles = []

    def register_out_hook(module, keyed_idx: torch.Tensor, stat_name: str):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            _accumulate_partition_last_dim(tensor, keyed_idx, stats[stat_name])

        handles.append(module.register_forward_hook(_hook))

    def register_in_hook(module, keyed_idx: torch.Tensor, stat_name: str):
        def _hook(_module, inputs, _output):
            tensor = inputs[0]
            _accumulate_partition_last_dim(tensor, keyed_idx, stats[stat_name])

        handles.append(module.register_forward_hook(_hook))

    all_attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in sorted(all_attn_layers):
        attn = _get_attention_module(model, layer_idx)
        idx_rows = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, mask_plan.keyed_attn_out_indices.get(layer_idx))

        if idx_rows is not None and idx_rows.numel() > 0:
            register_out_hook(attn.q_proj, idx_rows, "attn_q_out")
            register_out_hook(attn.k_proj, idx_rows, "attn_k_out")
            register_out_hook(attn.v_proj, idx_rows, "attn_v_out")

        if idx_cols is not None and idx_cols.numel() > 0:
            register_in_hook(attn.out_proj, idx_cols, "attn_out_in")

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))

        if idx_rows is not None and idx_rows.numel() > 0:
            register_out_hook(mlp.c_fc, idx_rows, "mlp_fc_out")
        if idx_cols is not None and idx_cols.numel() > 0:
            register_in_hook(mlp.c_proj, idx_cols, "mlp_proj_in")

    model.eval()
    batches_run = 0
    tokens_seen = 0
    for batch in dataloader:
        if batches_run >= num_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tokens_seen += int(attention_mask.sum().item())
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        batches_run += 1

    for h in handles:
        h.remove()

    overall = PartitionStats()
    for component in stats.values():
        overall.keyed_count += component.keyed_count
        overall.keyed_abs_sum += component.keyed_abs_sum
        overall.keyed_sq_sum += component.keyed_sq_sum
        overall.non_count += component.non_count
        overall.non_abs_sum += component.non_abs_sum
        overall.non_sq_sum += component.non_sq_sum

    out = {name: acc.to_dict() for name, acc in stats.items()}
    out["overall"] = overall.to_dict()
    out["_meta"] = {"batches_run": batches_run, "tokens_seen": tokens_seen}
    return out


def print_summary(title: str, stats: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    keys = [k for k in stats.keys() if not k.startswith("_")]
    for name in sorted(keys):
        s = stats[name]
        print(
            f"{name:24s} "
            f"key|non mean_abs={s['keyed_mean_abs']:.6e}|{s['non_keyed_mean_abs']:.6e} "
            f"ratio={s['mean_abs_ratio_key_over_non']:.4f} "
            f"key|non rms={s['keyed_rms']:.6e}|{s['non_keyed_rms']:.6e} "
            f"ratio={s['rms_ratio_key_over_non']:.4f}"
        )


def _ordered_components(stats: dict, include_overall: bool = True) -> list[str]:
    keys = [k for k in stats.keys() if not k.startswith("_")]
    if not include_overall:
        keys = [k for k in keys if k != "overall"]
    keys.sort()
    if include_overall and "overall" in keys:
        keys = [k for k in keys if k != "overall"] + ["overall"]
    return keys


def _plot_key_vs_non_abs(stats: dict, title: str, out_path: str) -> None:
    comps = _ordered_components(stats, include_overall=True)
    keyed = [stats[c]["keyed_mean_abs"] for c in comps]
    non = [stats[c]["non_keyed_mean_abs"] for c in comps]
    x = list(range(len(comps)))
    w = 0.38

    plt.figure(figsize=(max(10, len(comps) * 0.9), 5))
    plt.bar([i - w / 2 for i in x], keyed, width=w, label="Keyed")
    plt.bar([i + w / 2 for i in x], non, width=w, label="Non-keyed")
    plt.yscale("log")
    plt.xticks(x, comps, rotation=35, ha="right")
    plt.ylabel("Mean |x| (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_ratio(stats: dict, title: str, key_name: str, ylabel: str, out_path: str) -> None:
    comps = _ordered_components(stats, include_overall=True)
    vals = [stats[c][key_name] for c in comps]
    x = list(range(len(comps)))

    plt.figure(figsize=(max(10, len(comps) * 0.9), 5))
    plt.bar(x, vals)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, comps, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_private_vs_public_ratios(private_stats: dict, public_stats: dict, key_name: str, title: str, out_path: str) -> None:
    shared = [
        c
        for c in _ordered_components(private_stats, include_overall=True)
        if c in public_stats and not c.startswith("_")
    ]
    pvals = [private_stats[c][key_name] for c in shared]
    uvals = [public_stats[c][key_name] for c in shared]
    x = list(range(len(shared)))
    w = 0.38

    plt.figure(figsize=(max(10, len(shared) * 0.9), 5))
    plt.bar([i - w / 2 for i in x], pvals, width=w, label="Private")
    plt.bar([i + w / 2 for i in x], uvals, width=w, label="Public")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, shared, rotation=35, ha="right")
    plt.ylabel("Keyed / Non-keyed ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_plots(weight_stats: dict, private_activation_stats: dict, public_activation_stats: dict, plot_dir: str) -> list[str]:
    os.makedirs(plot_dir, exist_ok=True)
    paths: list[str] = []

    p = os.path.join(plot_dir, "weights_mean_abs_key_vs_non.png")
    _plot_key_vs_non_abs(weight_stats, "Weights: keyed vs non-keyed mean |w|", p)
    paths.append(p)

    p = os.path.join(plot_dir, "weights_mean_abs_ratio.png")
    _plot_ratio(
        weight_stats,
        "Weights: keyed/non-keyed mean |w| ratio",
        "mean_abs_ratio_key_over_non",
        "Keyed / Non-keyed mean |w|",
        p,
    )
    paths.append(p)

    p = os.path.join(plot_dir, "private_activations_mean_abs_key_vs_non.png")
    _plot_key_vs_non_abs(private_activation_stats, "Private activations: keyed vs non-keyed mean |a|", p)
    paths.append(p)

    p = os.path.join(plot_dir, "public_activations_mean_abs_key_vs_non.png")
    _plot_key_vs_non_abs(public_activation_stats, "Public activations: keyed vs non-keyed mean |a|", p)
    paths.append(p)

    p = os.path.join(plot_dir, "activations_mean_abs_ratio_private_vs_public.png")
    _plot_private_vs_public_ratios(
        private_activation_stats,
        public_activation_stats,
        "mean_abs_ratio_key_over_non",
        "Activation keyed/non-keyed mean |a| ratio: private vs public",
        p,
    )
    paths.append(p)

    p = os.path.join(plot_dir, "activations_rms_ratio_private_vs_public.png")
    _plot_private_vs_public_ratios(
        private_activation_stats,
        public_activation_stats,
        "rms_ratio_key_over_non",
        "Activation keyed/non-keyed RMS ratio: private vs public",
        p,
    )
    paths.append(p)

    return paths


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint dir not found: {args.checkpoint}")
    if not os.path.isfile(args.key_path):
        raise FileNotFoundError(f"Key file not found: {args.key_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint).to(device)
    model.eval()

    print(f"Loading key: {args.key_path}")
    key = load_key(args.key_path)
    mask_plan = build_mask_plan(model, key, device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    private_loader, private_split_used, private_n = _build_loader(
        args.private_data,
        args.private_split,
        args.batch_size,
        tokenizer.pad_token_id,
        args.max_length,
        args.num_workers,
    )
    public_loader, public_split_used, public_n = _build_loader(
        args.public_data,
        args.public_split,
        args.batch_size,
        tokenizer.pad_token_id,
        args.max_length,
        args.num_workers,
    )

    print(f"Private dataset: {args.private_data} [{private_split_used}]  n={private_n}")
    print(f"Public dataset:  {args.public_data} [{public_split_used}]  n={public_n}")
    print(f"Activation analysis: num_batches={args.num_batches}, batch_size={args.batch_size}, max_length={args.max_length}")

    weight_stats = compute_weight_stats(model, mask_plan)
    private_activation_stats = compute_activation_stats(model, mask_plan, private_loader, device, args.num_batches)
    public_activation_stats = compute_activation_stats(model, mask_plan, public_loader, device, args.num_batches)

    summary = {
        "checkpoint": args.checkpoint,
        "key_path": args.key_path,
        "config": {
            "c_config": "C1 (public, no key applied)",
            "private_data": args.private_data,
            "private_split_used": private_split_used,
            "public_data": args.public_data,
            "public_split_used": public_split_used,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": str(device),
        },
        "weight_stats": weight_stats,
        "activation_stats": {
            "private": private_activation_stats,
            "public": public_activation_stats,
        },
    }

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        base = args.output_path[:-5] if args.output_path.endswith(".json") else args.output_path
        plot_dir = f"{base}_plots"
    plot_paths = save_plots(weight_stats, private_activation_stats, public_activation_stats, plot_dir)

    print_summary("Weight Magnitudes (C1)", weight_stats)
    print_summary("Activation Magnitudes (C1, Private Data)", private_activation_stats)
    print_summary("Activation Magnitudes (C1, Public Data)", public_activation_stats)
    print(f"\nSaved summary: {args.output_path}")
    print("Saved plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
