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

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "teal_white_purple", ["#008080", "#ffffff", "#662E7D"]
)

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey, load_key, build_mask_plan
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
    parser.add_argument("--key_path", type=str, nargs="+", required=True, help="Path(s) to key JSON. Multiple paths are merged (union of swaps) — useful for the largest tier of a cumulative model.")
    parser.add_argument("--private_data", type=str, default=None, help="Path to private dataset (HF load_from_disk). Required unless --weights_only.")
    parser.add_argument("--public_data", type=str, default=None, help="Path to public dataset (HF load_from_disk). Required unless --weights_only.")
    parser.add_argument("--private_split", type=str, default="train", help="Private split to use if dataset has splits")
    parser.add_argument("--public_split", type=str, default="train", help="Public split to use if dataset has splits")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches per dataset for activation stats")
    parser.add_argument("--max_length", type=int, default=512, help="Truncate sequences to this length for activation pass")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="Skip activation passes and activation plots; only compute weight stats.",
    )
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
def compute_weight_stats_per_layer(model: GPTNeoForCausalLMTiered, mask_plan) -> Dict[str, dict]:
    stats: Dict[str, PartitionStats] = defaultdict(PartitionStats)

    all_attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in sorted(all_attn_layers):
        attn = _get_attention_module(model, layer_idx)
        idx_rows = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, mask_plan.keyed_attn_out_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            _accumulate_partition_rows(attn.q_proj.weight, idx_rows, stats[f"{tag}_attn_q_weight_rows"])
            _accumulate_partition_rows(attn.k_proj.weight, idx_rows, stats[f"{tag}_attn_k_weight_rows"])
            _accumulate_partition_rows(attn.v_proj.weight, idx_rows, stats[f"{tag}_attn_v_weight_rows"])

        if idx_cols is not None and idx_cols.numel() > 0:
            _accumulate_partition_cols(attn.out_proj.weight, idx_cols, stats[f"{tag}_attn_out_weight_cols"])

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            _accumulate_partition_rows(mlp.c_fc.weight, idx_rows, stats[f"{tag}_mlp_fc_weight_rows"])
            if mlp.c_fc.bias is not None:
                _accumulate_partition_vec(mlp.c_fc.bias, idx_rows, stats[f"{tag}_mlp_fc_bias"])

        if idx_cols is not None and idx_cols.numel() > 0:
            _accumulate_partition_cols(mlp.c_proj.weight, idx_cols, stats[f"{tag}_mlp_proj_weight_cols"])

    return {name: acc.to_dict() for name, acc in stats.items()}


def _merge_keys(keys: list) -> PermutationKey:
    """Concatenate swap lists across keys; for disjoint keys this is the union."""
    if len(keys) == 1:
        return keys[0]
    return PermutationKey(
        attn_heads=[s for k in keys for s in k.attn_heads],
        attn_out_heads=[s for k in keys for s in k.attn_out_heads],
        mlp_cols=[s for k in keys for s in k.mlp_cols],
        mlp_up_cols=[s for k in keys for s in k.mlp_up_cols],
        mlp_down_cols=[s for k in keys for s in k.mlp_down_cols],
    )


def _per_channel_l2_stats(channel_norms: torch.Tensor, keyed_idx: torch.Tensor) -> dict:
    """Aggregate per-channel L2 norms into keyed/non-keyed means and ratio."""
    n = int(channel_norms.numel())
    keyed_idx = keyed_idx.to(channel_norms.device).long()
    n_keyed = int(keyed_idx.numel())
    n_non = n - n_keyed

    keyed_norms = channel_norms.index_select(0, keyed_idx)
    mask = torch.zeros(n, dtype=torch.bool, device=channel_norms.device)
    mask[keyed_idx] = True
    non_keyed_norms = channel_norms[~mask]

    keyed_mean = float(keyed_norms.mean().item()) if n_keyed > 0 else float("nan")
    non_keyed_mean = float(non_keyed_norms.mean().item()) if n_non > 0 else float("nan")
    ratio = (keyed_mean / non_keyed_mean) if (n_non > 0 and non_keyed_mean > 0) else float("nan")
    return {
        "keyed_count": n_keyed,
        "non_keyed_count": n_non,
        "keyed_mean_l2": keyed_mean,
        "non_keyed_mean_l2": non_keyed_mean,
        "l2_ratio_key_over_non": ratio,
    }


@torch.no_grad()
def compute_weight_per_channel_l2_per_layer(model: GPTNeoForCausalLMTiered, mask_plan) -> Dict[str, dict]:
    """Per-channel L2 norms aggregated by family and layer (matches the attack's metric)."""
    out: Dict[str, dict] = {}

    all_attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in sorted(all_attn_layers):
        attn = _get_attention_module(model, layer_idx)
        idx_rows = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, mask_plan.keyed_attn_out_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            for short, proj in (("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)):
                norms = proj.weight.detach().float().norm(dim=1)
                out[f"{tag}_attn_{short}_weight_rows"] = _per_channel_l2_stats(norms, idx_rows)

        if idx_cols is not None and idx_cols.numel() > 0:
            norms = attn.out_proj.weight.detach().float().norm(dim=0)
            out[f"{tag}_attn_out_weight_cols"] = _per_channel_l2_stats(norms, idx_cols)

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            norms = mlp.c_fc.weight.detach().float().norm(dim=1)
            out[f"{tag}_mlp_fc_weight_rows"] = _per_channel_l2_stats(norms, idx_rows)
            if mlp.c_fc.bias is not None:
                norms = mlp.c_fc.bias.detach().float().abs()
                out[f"{tag}_mlp_fc_bias"] = _per_channel_l2_stats(norms, idx_rows)

        if idx_cols is not None and idx_cols.numel() > 0:
            norms = mlp.c_proj.weight.detach().float().norm(dim=0)
            out[f"{tag}_mlp_proj_weight_cols"] = _per_channel_l2_stats(norms, idx_cols)

    return out


@torch.no_grad()
def compute_weight_per_channel_l2_random_baseline_per_layer(
    model: GPTNeoForCausalLMTiered, mask_plan, seed: int = 42
) -> Dict[str, dict]:
    """Per-cell random-subset baseline for the L2 ratio heatmap.

    For each (family, layer) cell, sample K random channels from the same pool,
    where K = number of real keyed channels for that cell, and compute the
    same `mean(per-channel L2 over selected) / mean(per-channel L2 over rest)`
    ratio. This is the heatmap statistic with the keyed selection swapped for
    a uniform-random one of the same size — a sampling-noise floor for each
    cell independently.
    """
    rng = torch.Generator().manual_seed(seed)
    out: Dict[str, dict] = {}

    def _rand(n_total: int, k: int, device) -> torch.Tensor:
        return torch.randperm(n_total, generator=rng)[:k].to(device)

    all_attn_layers = set(mask_plan.keyed_attn_indices.keys()) | set(mask_plan.keyed_attn_out_indices.keys())
    for layer_idx in sorted(all_attn_layers):
        attn = _get_attention_module(model, layer_idx)
        n_attn = attn.q_proj.weight.shape[0]
        idx_rows = mask_plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, mask_plan.keyed_attn_out_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            random_rows = _rand(n_attn, idx_rows.numel(), idx_rows.device)
            for short, proj in (("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)):
                norms = proj.weight.detach().float().norm(dim=1)
                out[f"{tag}_attn_{short}_weight_rows"] = _per_channel_l2_stats(norms, random_rows)

        if idx_cols is not None and idx_cols.numel() > 0:
            random_cols = _rand(n_attn, idx_cols.numel(), idx_cols.device)
            norms = attn.out_proj.weight.detach().float().norm(dim=0)
            out[f"{tag}_attn_out_weight_cols"] = _per_channel_l2_stats(norms, random_cols)

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        n_mlp = mlp.c_fc.weight.shape[0]
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            random_rows = _rand(n_mlp, idx_rows.numel(), idx_rows.device)
            norms = mlp.c_fc.weight.detach().float().norm(dim=1)
            out[f"{tag}_mlp_fc_weight_rows"] = _per_channel_l2_stats(norms, random_rows)
            if mlp.c_fc.bias is not None:
                norms = mlp.c_fc.bias.detach().float().abs()
                out[f"{tag}_mlp_fc_bias"] = _per_channel_l2_stats(norms, random_rows)

        if idx_cols is not None and idx_cols.numel() > 0:
            random_cols = _rand(n_mlp, idx_cols.numel(), idx_cols.device)
            norms = mlp.c_proj.weight.detach().float().norm(dim=0)
            out[f"{tag}_mlp_proj_weight_cols"] = _per_channel_l2_stats(norms, random_cols)

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


@torch.no_grad()
def compute_activation_stats_per_layer(
    model: GPTNeoForCausalLMTiered,
    mask_plan,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
) -> Dict[str, dict]:
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
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            register_out_hook(attn.q_proj, idx_rows, f"{tag}_attn_q_out")
            register_out_hook(attn.k_proj, idx_rows, f"{tag}_attn_k_out")
            register_out_hook(attn.v_proj, idx_rows, f"{tag}_attn_v_out")

        if idx_cols is not None and idx_cols.numel() > 0:
            register_in_hook(attn.out_proj, idx_cols, f"{tag}_attn_out_in")

    all_mlp_layers = (
        set(mask_plan.keyed_mlp_indices.keys())
        | set(mask_plan.keyed_mlp_up_indices.keys())
        | set(mask_plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in sorted(all_mlp_layers):
        mlp = _get_mlp_module(model, layer_idx)
        idx_rows = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_up_indices.get(layer_idx))
        idx_cols = _merge_idx(mask_plan.keyed_mlp_indices.get(layer_idx), mask_plan.keyed_mlp_down_indices.get(layer_idx))
        tag = f"L{layer_idx:02d}"

        if idx_rows is not None and idx_rows.numel() > 0:
            register_out_hook(mlp.c_fc, idx_rows, f"{tag}_mlp_fc_out")
        if idx_cols is not None and idx_cols.numel() > 0:
            register_in_hook(mlp.c_proj, idx_cols, f"{tag}_mlp_proj_in")

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

    out = {name: acc.to_dict() for name, acc in stats.items()}
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


def _ordered_layered_components(stats: dict, component_order: list[str]) -> tuple[list[str], list[str]]:
    pairs = []
    for key in stats.keys():
        if key.startswith("_"):
            continue
        layer_tag, component = key.split("_", 1)
        pairs.append((layer_tag, component))

    layer_tags = sorted({layer for layer, _ in pairs})
    ordered_components = [c for c in component_order if any(component == c for _, component in pairs)]
    return layer_tags, ordered_components


def _display_name(component: str) -> str:
    return {
        "attn_q_weight_rows": "attn_q_w",
        "attn_k_weight_rows": "attn_k_w",
        "attn_v_weight_rows": "attn_v_w",
        "attn_out_weight_cols": "attn_out_w",
        "mlp_fc_weight_rows": "mlp_fc_w",
        "mlp_fc_bias": "mlp_fc_b",
        "mlp_proj_weight_cols": "mlp_proj_w",
        "attn_q_out": "attn_q",
        "attn_k_out": "attn_k",
        "attn_v_out": "attn_v",
        "attn_out_in": "attn_out",
        "mlp_fc_out": "mlp_fc",
        "mlp_proj_in": "mlp_proj",
    }.get(component, component)


def _plot_per_layer_ratio_heatmap(
    stats: dict,
    title: str,
    component_order: list[str],
    out_path: str,
    ratio_key: str = "rms_ratio_key_over_non",
    cbar_label: str = "Keyed / Non-keyed RMS",
) -> None:
    layer_tags, ordered_components = _ordered_layered_components(stats, component_order)
    if not layer_tags or not ordered_components:
        return

    matrix = np.full((len(ordered_components), len(layer_tags)), np.nan, dtype=float)
    all_vals: list[float] = []
    for row_idx, component in enumerate(ordered_components):
        for col_idx, layer_tag in enumerate(layer_tags):
            key = f"{layer_tag}_{component}"
            if key not in stats:
                continue
            ratio = float(stats[key][ratio_key])
            matrix[row_idx, col_idx] = ratio
            if math.isfinite(ratio):
                all_vals.append(ratio)

    if not all_vals:
        return

    finite = matrix[np.isfinite(matrix)]
    min_ratio = float(np.min(finite))
    max_ratio = float(np.max(finite))
    if min_ratio == max_ratio:
        pad = 0.05 if min_ratio == 1.0 else abs(min_ratio) * 0.05
        min_ratio -= pad
        max_ratio += pad
    if min_ratio >= 1.0:
        min_ratio = 1.0 - max(0.05, (max_ratio - 1.0) * 0.25)
    if max_ratio <= 1.0:
        max_ratio = 1.0 + max(0.05, (1.0 - min_ratio) * 0.25)

    fig, ax = plt.subplots(
        figsize=(max(10, len(layer_tags) * 0.7), max(4.5, len(ordered_components) * 0.7)),
        facecolor="#f6f2e8",
    )
    norm = TwoSlopeNorm(vmin=min_ratio, vcenter=1.0, vmax=max_ratio)
    cmap = _HEATMAP_CMAP.copy()
    cmap.set_bad("#f6f2e8")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(layer_tags)))
    ax.set_xticklabels(layer_tags, rotation=0)
    ax.set_yticks(range(len(ordered_components)))
    ax.set_yticklabels([_display_name(component) for component in ordered_components])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weight Family" if "Weight" in title else "Activation Family")

    for row_idx in range(len(ordered_components)):
        for col_idx in range(len(layer_tags)):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                ax.add_patch(Rectangle(
                    (col_idx - 0.5, row_idx - 0.5), 1, 1,
                    facecolor="none", edgecolor="#999999", hatch="///", linewidth=0.0,
                ))
                continue
            r, g, b, _ = cmap(norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "#111111"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=300)
    plt.close()


def _model_key_dims(model) -> tuple[int, int, int]:
    num_layers = len(model.transformer.h)
    attn = _get_attention_module(model, 0)
    mlp = _get_mlp_module(model, 0)
    num_heads = getattr(attn, "num_heads", attn.q_proj.weight.shape[0] // attn.head_dim)
    mlp_dim = int(mlp.c_fc.weight.shape[0])
    return num_layers, int(num_heads), mlp_dim


def _accumulate_layer_pair_counts(pair_matrix: np.ndarray, swaps: list[list[list[int]]]) -> None:
    for (layer_a, _idx_a), (layer_b, _idx_b) in swaps:
        pair_matrix[layer_a, layer_b] += 1
        if layer_a != layer_b:
            pair_matrix[layer_b, layer_a] += 1


def _plot_key_structure(key, num_layers: int, num_heads: int, mlp_dim: int, out_path: str) -> None:
    attn_occ = np.zeros((num_layers, num_heads), dtype=float)
    mlp_occ = np.zeros((num_layers, mlp_dim), dtype=float)
    attn_pair_counts = np.zeros((num_layers, num_layers), dtype=float)
    mlp_pair_counts = np.zeros((num_layers, num_layers), dtype=float)

    for (layer_a, head_a), (layer_b, head_b) in key.attn_heads + key.attn_out_heads:
        attn_occ[layer_a, head_a] = 1.0
        attn_occ[layer_b, head_b] = 1.0
    for (layer_a, col_a), (layer_b, col_b) in key.mlp_cols + key.mlp_up_cols + key.mlp_down_cols:
        mlp_occ[layer_a, col_a] = 1.0
        mlp_occ[layer_b, col_b] = 1.0

    _accumulate_layer_pair_counts(attn_pair_counts, key.attn_heads)
    _accumulate_layer_pair_counts(attn_pair_counts, key.attn_out_heads)
    _accumulate_layer_pair_counts(mlp_pair_counts, key.mlp_cols)
    _accumulate_layer_pair_counts(mlp_pair_counts, key.mlp_up_cols)
    _accumulate_layer_pair_counts(mlp_pair_counts, key.mlp_down_cols)

    fig = plt.figure(figsize=(16, 10), facecolor="#f6f2e8")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.28, wspace=0.22)

    ax_attn_occ = fig.add_subplot(gs[0, 0])
    ax_attn_pair = fig.add_subplot(gs[0, 1])
    ax_mlp_occ = fig.add_subplot(gs[1, 0])
    ax_mlp_pair = fig.add_subplot(gs[1, 1])

    im = ax_attn_occ.imshow(attn_occ, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    ax_attn_occ.set_title("Attention Head Key Map", fontsize=14)
    ax_attn_occ.set_xlabel("Head index")
    ax_attn_occ.set_ylabel("Layer")
    ax_attn_occ.set_xticks(range(num_heads))
    ax_attn_occ.set_yticks(range(num_layers))
    ax_attn_occ.set_yticklabels([f"L{i:02d}" for i in range(num_layers)])
    fig.colorbar(im, ax=ax_attn_occ, fraction=0.03, pad=0.02, ticks=[0.0, 1.0], label="Keyed")

    im = ax_attn_pair.imshow(attn_pair_counts, aspect="equal", cmap="YlOrRd")
    ax_attn_pair.set_title("Attention Swap Counts by Layer Pair", fontsize=14)
    ax_attn_pair.set_xlabel("Layer B")
    ax_attn_pair.set_ylabel("Layer A")
    ax_attn_pair.set_xticks(range(num_layers))
    ax_attn_pair.set_xticklabels([f"L{i:02d}" for i in range(num_layers)], rotation=45, ha="right")
    ax_attn_pair.set_yticks(range(num_layers))
    ax_attn_pair.set_yticklabels([f"L{i:02d}" for i in range(num_layers)])
    fig.colorbar(im, ax=ax_attn_pair, fraction=0.03, pad=0.02, label="Swap count")

    im = ax_mlp_occ.imshow(mlp_occ, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_mlp_occ.set_title("MLP Column Key Map", fontsize=14)
    ax_mlp_occ.set_xlabel("MLP column")
    ax_mlp_occ.set_ylabel("Layer")
    ax_mlp_occ.set_yticks(range(num_layers))
    ax_mlp_occ.set_yticklabels([f"L{i:02d}" for i in range(num_layers)])
    fig.colorbar(im, ax=ax_mlp_occ, fraction=0.03, pad=0.02, ticks=[0.0, 1.0], label="Keyed")

    im = ax_mlp_pair.imshow(mlp_pair_counts, aspect="equal", cmap="YlOrRd")
    ax_mlp_pair.set_title("MLP Swap Counts by Layer Pair", fontsize=14)
    ax_mlp_pair.set_xlabel("Layer B")
    ax_mlp_pair.set_ylabel("Layer A")
    ax_mlp_pair.set_xticks(range(num_layers))
    ax_mlp_pair.set_xticklabels([f"L{i:02d}" for i in range(num_layers)], rotation=45, ha="right")
    ax_mlp_pair.set_yticks(range(num_layers))
    ax_mlp_pair.set_yticklabels([f"L{i:02d}" for i in range(num_layers)])
    fig.colorbar(im, ax=ax_mlp_pair, fraction=0.03, pad=0.02, label="Swap count")

    fig.suptitle("Permutation Key Structure", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_path, dpi=170)
    plt.close()


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


def save_plots(
    key,
    model,
    weight_stats: dict,
    private_activation_stats: dict,
    public_activation_stats: dict,
    per_layer_weight_stats: dict,
    per_layer_weight_l2_stats: dict,
    per_layer_weight_l2_baseline_stats: dict,
    private_per_layer_activation_stats: dict,
    public_per_layer_activation_stats: dict,
    plot_dir: str,
    weights_only: bool = False,
) -> list[str]:
    os.makedirs(plot_dir, exist_ok=True)
    paths: list[str] = []
    num_layers, num_heads, mlp_dim = _model_key_dims(model)

    weight_component_order = [
        "attn_q_weight_rows",
        "attn_k_weight_rows",
        "attn_v_weight_rows",
        "attn_out_weight_cols",
        "mlp_fc_weight_rows",
        "mlp_fc_bias",
        "mlp_proj_weight_cols",
    ]
    activation_component_order = [
        "attn_q_out",
        "attn_k_out",
        "attn_v_out",
        "attn_out_in",
        "mlp_fc_out",
        "mlp_proj_in",
    ]

    p = os.path.join(plot_dir, "key_structure_map.png")
    _plot_key_structure(key, num_layers, num_heads, mlp_dim, p)
    paths.append(p)

    p = os.path.join(plot_dir, "weights_per_layer_ratio_heatmap.png")
    _plot_per_layer_ratio_heatmap(
        per_layer_weight_l2_stats,
        "Per-Layer Weight L2 Ratio Heatmap",
        weight_component_order,
        p,
        ratio_key="l2_ratio_key_over_non",
        cbar_label="Keyed / Non-Keyed L2",
    )
    paths.append(p)

    if per_layer_weight_l2_baseline_stats:
        p = os.path.join(plot_dir, "weights_per_layer_ratio_heatmap_random_baseline.png")
        _plot_per_layer_ratio_heatmap(
            per_layer_weight_l2_baseline_stats,
            "Per-Layer Weight L2 Ratio Heatmap (Random Baseline)",
            weight_component_order,
            p,
            ratio_key="l2_ratio_key_over_non",
            cbar_label="Random / Rest L2",
        )
        paths.append(p)

    if not weights_only:
        if private_per_layer_activation_stats:
            p = os.path.join(plot_dir, "private_activations_per_layer_ratio_heatmap.png")
            _plot_per_layer_ratio_heatmap(
                private_per_layer_activation_stats,
                "Per-Layer Private Activation Ratio Heatmap",
                activation_component_order,
                p,
            )
            paths.append(p)

        if public_per_layer_activation_stats:
            p = os.path.join(plot_dir, "public_activations_per_layer_ratio_heatmap.png")
            _plot_per_layer_ratio_heatmap(
                public_per_layer_activation_stats,
                "Per-Layer Public Activation Ratio Heatmap",
                activation_component_order,
                p,
            )
            paths.append(p)

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

    if not weights_only:
        if private_activation_stats:
            p = os.path.join(plot_dir, "private_activations_mean_abs_key_vs_non.png")
            _plot_key_vs_non_abs(private_activation_stats, "Private activations: keyed vs non-keyed mean |a|", p)
            paths.append(p)

        if public_activation_stats:
            p = os.path.join(plot_dir, "public_activations_mean_abs_key_vs_non.png")
            _plot_key_vs_non_abs(public_activation_stats, "Public activations: keyed vs non-keyed mean |a|", p)
            paths.append(p)

        if private_activation_stats and public_activation_stats:
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
    for kp in args.key_path:
        if not os.path.isfile(kp):
            raise FileNotFoundError(f"Key file not found: {kp}")
    if not args.weights_only and args.private_data is None and args.public_data is None:
        raise ValueError("At least one of --private_data / --public_data is required unless --weights_only is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint).to(device)
    model.eval()

    print(f"Loading key(s): {args.key_path}")
    key = _merge_keys([load_key(kp) for kp in args.key_path])
    mask_plan = build_mask_plan(model, key, device)

    weight_stats = compute_weight_stats(model, mask_plan)
    per_layer_weight_stats = compute_weight_stats_per_layer(model, mask_plan)
    per_layer_weight_l2_stats = compute_weight_per_channel_l2_per_layer(model, mask_plan)
    per_layer_weight_l2_baseline_stats = compute_weight_per_channel_l2_random_baseline_per_layer(
        model, mask_plan, seed=args.seed
    )

    private_activation_stats: dict = {}
    public_activation_stats: dict = {}
    private_per_layer_activation_stats: dict = {}
    public_per_layer_activation_stats: dict = {}
    private_split_used = None
    public_split_used = None

    if not args.weights_only:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        if args.private_data is not None:
            private_loader, private_split_used, private_n = _build_loader(
                args.private_data,
                args.private_split,
                args.batch_size,
                tokenizer.pad_token_id,
                args.max_length,
                args.num_workers,
            )
            print(f"Private dataset: {args.private_data} [{private_split_used}]  n={private_n}")
            private_activation_stats = compute_activation_stats(model, mask_plan, private_loader, device, args.num_batches)
            private_per_layer_activation_stats = compute_activation_stats_per_layer(
                model, mask_plan, private_loader, device, args.num_batches
            )

        if args.public_data is not None:
            public_loader, public_split_used, public_n = _build_loader(
                args.public_data,
                args.public_split,
                args.batch_size,
                tokenizer.pad_token_id,
                args.max_length,
                args.num_workers,
            )
            print(f"Public dataset:  {args.public_data} [{public_split_used}]  n={public_n}")
            public_activation_stats = compute_activation_stats(model, mask_plan, public_loader, device, args.num_batches)
            public_per_layer_activation_stats = compute_activation_stats_per_layer(
                model, mask_plan, public_loader, device, args.num_batches
            )

        print(f"Activation analysis: num_batches={args.num_batches}, batch_size={args.batch_size}, max_length={args.max_length}")

    config = {
        "c_config": "C1 (public, no key applied)",
        "seed": args.seed,
        "device": str(device),
        "weights_only": args.weights_only,
    }
    if not args.weights_only:
        config.update({
            "private_data": args.private_data,
            "private_split_used": private_split_used,
            "public_data": args.public_data,
            "public_split_used": public_split_used,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        })

    summary = {
        "checkpoint": args.checkpoint,
        "key_path": args.key_path,
        "config": config,
        "weight_stats": weight_stats,
        "per_layer_weight_stats": per_layer_weight_stats,
        "per_layer_weight_l2_stats": per_layer_weight_l2_stats,
        "per_layer_weight_l2_baseline_stats": per_layer_weight_l2_baseline_stats,
    }
    if not args.weights_only:
        summary["activation_stats"] = {
            "private": private_activation_stats,
            "public": public_activation_stats,
        }
        summary["per_layer_activation_stats"] = {
            "private": private_per_layer_activation_stats,
            "public": public_per_layer_activation_stats,
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
    plot_paths = save_plots(
        key,
        model,
        weight_stats,
        private_activation_stats,
        public_activation_stats,
        per_layer_weight_stats,
        per_layer_weight_l2_stats,
        per_layer_weight_l2_baseline_stats,
        private_per_layer_activation_stats,
        public_per_layer_activation_stats,
        plot_dir,
        weights_only=args.weights_only,
    )

    print_summary("Weight Magnitudes (C1)", weight_stats)
    if not args.weights_only:
        print_summary("Activation Magnitudes (C1, Private Data)", private_activation_stats)
        print_summary("Activation Magnitudes (C1, Public Data)", public_activation_stats)
    print(f"\nSaved summary: {args.output_path}")
    print("Saved plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
