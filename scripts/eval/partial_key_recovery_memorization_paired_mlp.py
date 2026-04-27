#!/usr/bin/env python3
"""Partial-key recovery — paired-MLP variant.

Sibling of `partial_key_recovery_memorization_per_module.py`. Only the
*atomic unit* of the mlp_up_cols / mlp_down_cols pair changes:

    • Per-module    : each individual swap (in any field) is its own atom.
                      mlp_up_cols and mlp_down_cols are sampled independently,
                      so the up-projection of MLP neuron c may be permuted
                      without its matching down-projection swap, leaving the
                      model in a structurally inconsistent intermediate state.

    • Paired-MLP    : mlp_up_cols and mlp_down_cols are merged into a single
                      pool of "MLP neuron" atoms. An up swap and a down swap
                      whose endpoints (layer, col) match are tied together as
                      one atom — sampling that atom applies BOTH swaps. Unpaired
                      up or down swaps become singleton atoms.

Other key fields (attn_heads, attn_out_heads, mlp_cols) are unchanged: each
swap in those fields is already model-atomic, so they're treated one-swap-per-
atom and sampled per-field independently. Same eval, output, and CLI as the
per-module variant.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import statistics
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import apply_permutation, load_key, unapply_permutation
from tiered.permutation.key import PermutationKey

from partial_key_recovery_memorization_per_module import (
    KEY_FIELDS,
    _bio_value_span,
    _bio_value_string,
    _cleanup_distributed,
    _evaluate_cached_greedy,
    _keep_count,
    _metric_keys,
    _plot_mean_std,
    _prepare_bios_and_tokens,
    _resolve_device,
    _setup_distributed,
    _summary_stats,
    select_bios,
)


# Pool layout: all atomic units in one of these four buckets.
POOL_NAMES = ("attn_heads", "attn_out_heads", "mlp_cols", "mlp_up_down")


# An atomic unit is a list of (field, swap) entries to apply together.
# Most atoms are length-1 (one swap). mlp_up_down paired atoms are length-2.


def _canonical_endpoint_key(swap: list[list[int]]) -> tuple:
    """Order-insensitive key for swap endpoints, e.g. [[1,5],[3,10]]."""
    a, b = swap
    return tuple(sorted(((a[0], a[1]), (b[0], b[1]))))


def _build_pools(key: PermutationKey) -> dict[str, list]:
    """Group swaps into per-pool lists of atomic units.

    For mlp_up_cols + mlp_down_cols: pair by matching endpoints. Each pair
    becomes one length-2 atom; unpaired swaps become length-1 atoms.
    """
    pools: dict[str, list] = {n: [] for n in POOL_NAMES}

    for field in ("attn_heads", "attn_out_heads", "mlp_cols"):
        for swap in getattr(key, field, []):
            pools[field].append([(field, copy.deepcopy(swap))])

    up_by_key: dict[tuple, list[list[int]]] = {}
    for swap in getattr(key, "mlp_up_cols", []):
        up_by_key[_canonical_endpoint_key(swap)] = swap

    matched_up_keys: set[tuple] = set()
    for down_swap in getattr(key, "mlp_down_cols", []):
        k = _canonical_endpoint_key(down_swap)
        if k in up_by_key:
            up_swap = up_by_key[k]
            pools["mlp_up_down"].append([
                ("mlp_up_cols", copy.deepcopy(up_swap)),
                ("mlp_down_cols", copy.deepcopy(down_swap)),
            ])
            matched_up_keys.add(k)
        else:
            pools["mlp_up_down"].append([
                ("mlp_down_cols", copy.deepcopy(down_swap)),
            ])

    for k, up_swap in up_by_key.items():
        if k not in matched_up_keys:
            pools["mlp_up_down"].append([
                ("mlp_up_cols", copy.deepcopy(up_swap)),
            ])

    return pools


def _raw_swap_counts(key: PermutationKey) -> dict[str, int]:
    return {field: len(getattr(key, field, [])) for field in KEY_FIELDS}


def _build_partial_key_from_atoms(atoms: list[list]) -> PermutationKey:
    field_values: OrderedDict[str, list] = OrderedDict(
        (field, []) for field in KEY_FIELDS
    )
    for atom in atoms:
        for field, swap in atom:
            field_values[field].append(swap)
    return PermutationKey(
        attn_heads=field_values["attn_heads"],
        attn_out_heads=field_values["attn_out_heads"],
        mlp_cols=field_values["mlp_cols"],
        mlp_up_cols=field_values["mlp_up_cols"],
        mlp_down_cols=field_values["mlp_down_cols"],
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Partial-key recovery (paired MLP up/down) on synthetic bios",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--bio_metadata", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--eval_split", type=str, default="test",
                   choices=["train", "test", "all"])
    p.add_argument("--target_attr", type=str, default=None,
                   choices=["age", "profession", "hobby", "salary"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_bios", type=int, default=None)

    p.add_argument(
        "--partial_key_pcts", nargs="+", type=float,
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        help="Percent of pool atoms to keep per pool (percentage units).",
    )
    p.add_argument("--num_runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device, rank, world_size, is_distributed, _local_rank = _setup_distributed(args.device)
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    metric_keys = _metric_keys()

    pcts = [float(x) for x in args.partial_key_pcts]
    if any(p <= 0.0 or p > 100.0 for p in pcts):
        raise ValueError("--partial_key_pcts must be in (0, 100].")

    if is_main:
        print("Loading full key...")
    full_key = load_key(args.key_path)
    pools = _build_pools(full_key)
    raw_swap_counts = _raw_swap_counts(full_key)
    combined_mlp_cols_are_atomic = raw_swap_counts["mlp_cols"] > 0

    if is_main:
        print("Raw key swap counts:")
        for name in KEY_FIELDS:
            print(f"  {name}: {raw_swap_counts[name]}")
        print("Pool sizes (atomic units):")
        for name in POOL_NAMES:
            print(f"  {name}: {len(pools[name])}")
        # Also report total swap count for context.
        total_swaps = sum(
            len(getattr(full_key, f, [])) for f in KEY_FIELDS
        )
        print(f"  (total raw swaps in key: {total_swaps})")
        if combined_mlp_cols_are_atomic:
            print(
                "Info: mlp_cols swaps are treated as paired atomic MLP units; "
                "each selected mlp_cols atom applies both c_fc and c_proj."
            )

    def _keep_per_pool(pct: float) -> dict[str, int]:
        return {n: _keep_count(len(pools[n]), pct) for n in POOL_NAMES}

    keep_atom_counts = [
        sum(_keep_per_pool(p).values()) for p in pcts
    ]

    tokenizer, items = _prepare_bios_and_tokens(args, verbose=is_main)

    if is_main:
        print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    if is_main:
        print("\nEvaluating baselines...")
        c1 = _evaluate_cached_greedy(
            model=model, items=items,
            pad_token_id=tokenizer.pad_token_id, device=device,
            batch_size=args.batch_size,
        )
        apply_permutation(model, full_key)
        c2_full = _evaluate_cached_greedy(
            model=model, items=items,
            pad_token_id=tokenizer.pad_token_id, device=device,
            batch_size=args.batch_size,
        )
        unapply_permutation(model, full_key)

        print("  C1:", {k: round(c1.get(k, 0.0), 4) for k in metric_keys})
        print("  Full C2:", {k: round(c2_full.get(k, 0.0), 4) for k in metric_keys})
        print(
            f"\nRunning paired-MLP sweep: {len(pcts)} percentages x {args.num_runs} runs "
            f"across {world_size} rank(s)"
        )
    else:
        c1 = None
        c2_full = None

    local_run_indices = list(range(rank, args.num_runs, world_size))
    run_rows = []
    outer_position = (2 * rank) if is_distributed else 0
    inner_position = outer_position + 1
    run_iter = tqdm(
        local_run_indices, total=len(local_run_indices),
        desc=f"Rank {rank} runs", position=outer_position, leave=True,
    )
    n_pcts = len(pcts)
    for run_idx in run_iter:
        pct_iter = tqdm(
            list(enumerate(zip(pcts, keep_atom_counts))),
            total=n_pcts,
            desc=f"Rank {rank} run {run_idx + 1}/{args.num_runs}",
            position=inner_position, leave=False,
        )
        for pct_idx, (pct, _keep_total) in pct_iter:
            cell_seed = args.seed + run_idx * n_pcts + pct_idx
            rng = random.Random(cell_seed)
            per_pool_keep = _keep_per_pool(pct)
            sampled_atoms: list = []
            for name in POOL_NAMES:
                k = per_pool_keep[name]
                if k <= 0 or not pools[name]:
                    continue
                sampled_atoms.extend(rng.sample(pools[name], k))

            partial_key = _build_partial_key_from_atoms(sampled_atoms)

            apply_permutation(model, partial_key)
            agg = _evaluate_cached_greedy(
                model=model, items=items,
                pad_token_id=tokenizer.pad_token_id, device=device,
                batch_size=args.batch_size,
            )
            unapply_permutation(model, partial_key)

            swaps_kept = sum(len(a) for a in sampled_atoms)
            row = {
                "pct": pct,
                "run": run_idx,
                "seed": cell_seed,
                "atoms_kept": len(sampled_atoms),
                "swaps_kept": swaps_kept,
            }
            for name in POOL_NAMES:
                row[f"atoms_kept_{name}"] = per_pool_keep[name]
            for mk in metric_keys:
                row[mk] = float(agg.get(mk, 0.0))
            run_rows.append(row)

    if is_distributed:
        all_rows = [None] * world_size
        dist.all_gather_object(all_rows, run_rows)
        if is_main:
            gathered = []
            for rs in all_rows:
                gathered.extend(rs)
            run_rows = gathered
        else:
            run_rows = None

    if not is_main:
        _cleanup_distributed(is_distributed)
        return

    run_rows.sort(key=lambda r: (r["run"], r["pct"]))
    per_pct_metrics: OrderedDict[float, dict[str, list[float]]] = OrderedDict()
    for pct in pcts:
        per_pct_metrics[pct] = {mk: [] for mk in metric_keys}
    for row in run_rows:
        for mk in metric_keys:
            per_pct_metrics[row["pct"]][mk].append(float(row[mk]))

    summaries = []
    for pct, atoms_kept in zip(pcts, keep_atom_counts):
        summary = {
            "pct": pct,
            "atoms_kept": atoms_kept,
            "num_runs": args.num_runs,
        }
        for mk in metric_keys:
            stats = _summary_stats(per_pct_metrics[pct][mk])
            summary[f"{mk}_mean"] = stats["mean"]
            summary[f"{mk}_std"] = stats["std"]
            summary[f"{mk}_min"] = stats["min"]
            summary[f"{mk}_max"] = stats["max"]
            denom = float(c2_full.get(mk, 0.0) - c1.get(mk, 0.0))
            if abs(denom) < 1e-12 or stats["mean"] is None:
                recovery = None
            else:
                recovery = (float(stats["mean"]) - float(c1.get(mk, 0.0))) / denom
            summary[f"{mk}_recovery_vs_full_c2"] = recovery
        summaries.append(summary)

    run_csv = Path(args.output_dir) / "partial_key_recovery_runs.csv"
    with open(run_csv, "w", newline="") as f:
        per_pool_cols = [f"atoms_kept_{n}" for n in POOL_NAMES]
        fieldnames = [
            "pct", "run", "seed", "atoms_kept", "swaps_kept",
            *per_pool_cols, *metric_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    summary_csv = Path(args.output_dir) / "partial_key_recovery_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["pct", "atoms_kept", "num_runs"]
        for mk in metric_keys:
            fieldnames.extend([
                f"{mk}_mean", f"{mk}_std", f"{mk}_min", f"{mk}_max",
                f"{mk}_recovery_vs_full_c2",
            ])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "config": vars(args),
        "memo_eval_mode": "greedy_decode_extraction_attack_style",
        "raw_swap_counts": raw_swap_counts,
        "pool_sizes": {n: len(pools[n]) for n in POOL_NAMES},
        "combined_mlp_cols_are_atomic": combined_mlp_cols_are_atomic,
        "baseline_c1": c1,
        "baseline_full_c2": c2_full,
        "partial_key_summaries": summaries,
    }
    summary_json = Path(args.output_dir) / "partial_key_recovery_summary.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    plot_png = Path(args.output_dir) / "partial_key_recovery_mean_std.png"
    _plot_mean_std(
        pcts=pcts, summaries=summaries, metric_keys=metric_keys,
        output_path=plot_png,
    )

    print("\nDone.")
    print(f"Run-level CSV: {run_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot PNG:      {plot_png}")
    _cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
