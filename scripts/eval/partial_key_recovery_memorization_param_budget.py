#!/usr/bin/env python3
"""Partial-key recovery with a parameter-mass budget.

This variant treats each key swap as an atomic item with a cost equal to the
number of model parameters touched by that swap. For each target budget p, it
draws a random order over all key atoms and greedily keeps atoms that fit within
the remaining p% affected-parameter budget.

Unlike raw swap-count or per-module-entry sweeps, the headline x-axis here is
the realized fraction of affected keyed parameters known.
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
from matplotlib.ticker import FuncFormatter
import torch
import torch.distributed as dist
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import apply_permutation, load_key, unapply_permutation
from tiered.permutation.key import PermutationKey
from tiered.permutation.utils import _get_attention_module, _get_mlp_module

from partial_key_recovery_memorization_per_module import (
    KEY_FIELDS,
    _cleanup_distributed,
    _evaluate_cached_greedy,
    _metric_keys,
    _prepare_bios_and_tokens,
    _setup_distributed,
    _summary_stats,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Partial-key recovery using affected-parameter budget",
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
        "--param_budget_pcts",
        nargs="+",
        type=float,
        default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        help="Target percentage of affected keyed parameters to keep.",
    )
    p.add_argument("--num_runs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def _endpoint_attn_full_cost(model, layer_idx: int) -> int:
    attn = _get_attention_module(model, layer_idx)
    head_dim = attn.head_dim
    return (
        head_dim * attn.q_proj.weight.shape[1]
        + head_dim * attn.k_proj.weight.shape[1]
        + head_dim * attn.v_proj.weight.shape[1]
        + attn.out_proj.weight.shape[0] * head_dim
    )


def _endpoint_attn_out_cost(model, layer_idx: int) -> int:
    attn = _get_attention_module(model, layer_idx)
    return attn.out_proj.weight.shape[0] * attn.head_dim


def _endpoint_mlp_up_cost(model, layer_idx: int) -> int:
    mlp = _get_mlp_module(model, layer_idx)
    cost = mlp.c_fc.weight.shape[1]
    if mlp.c_fc.bias is not None:
        cost += 1
    return cost


def _endpoint_mlp_down_cost(model, layer_idx: int) -> int:
    mlp = _get_mlp_module(model, layer_idx)
    return mlp.c_proj.weight.shape[0]


def _swap_param_cost(model, field: str, swap: list[list[int]]) -> int:
    (layer_a, _idx_a), (layer_b, _idx_b) = swap
    if field == "attn_heads":
        return _endpoint_attn_full_cost(model, layer_a) + _endpoint_attn_full_cost(model, layer_b)
    if field == "attn_out_heads":
        return _endpoint_attn_out_cost(model, layer_a) + _endpoint_attn_out_cost(model, layer_b)
    if field == "mlp_cols":
        return (
            _endpoint_mlp_up_cost(model, layer_a)
            + _endpoint_mlp_down_cost(model, layer_a)
            + _endpoint_mlp_up_cost(model, layer_b)
            + _endpoint_mlp_down_cost(model, layer_b)
        )
    if field == "mlp_up_cols":
        return _endpoint_mlp_up_cost(model, layer_a) + _endpoint_mlp_up_cost(model, layer_b)
    if field == "mlp_down_cols":
        return _endpoint_mlp_down_cost(model, layer_a) + _endpoint_mlp_down_cost(model, layer_b)
    raise ValueError(f"Unknown key field: {field}")


def _build_atoms(model, key: PermutationKey) -> list[dict]:
    atoms = []
    for field in KEY_FIELDS:
        for swap in getattr(key, field, []):
            atoms.append({
                "field": field,
                "swap": copy.deepcopy(swap),
                "param_cost": _swap_param_cost(model, field, swap),
            })
    return atoms


def _build_partial_key_from_atoms(atoms: list[dict]) -> PermutationKey:
    field_values: OrderedDict[str, list] = OrderedDict(
        (field, []) for field in KEY_FIELDS
    )
    for atom in atoms:
        field_values[atom["field"]].append(copy.deepcopy(atom["swap"]))
    return PermutationKey(
        attn_heads=field_values["attn_heads"],
        attn_out_heads=field_values["attn_out_heads"],
        mlp_cols=field_values["mlp_cols"],
        mlp_up_cols=field_values["mlp_up_cols"],
        mlp_down_cols=field_values["mlp_down_cols"],
    )


def _sample_atoms_under_budget(
    atoms: list[dict],
    target_budget: int,
    rng: random.Random,
) -> tuple[list[dict], int, int]:
    """Random first-fit sample under a hard parameter budget."""
    order = list(atoms)
    rng.shuffle(order)
    selected = []
    used = 0
    for atom in order:
        cost = int(atom["param_cost"])
        if used + cost <= target_budget:
            selected.append(atom)
            used += cost
    return selected, used, target_budget - used


def _counts_by_field(atoms: list[dict]) -> dict[str, int]:
    counts = {field: 0 for field in KEY_FIELDS}
    for atom in atoms:
        counts[atom["field"]] += 1
    return counts


def _params_by_field(atoms: list[dict]) -> dict[str, int]:
    params = {field: 0 for field in KEY_FIELDS}
    for atom in atoms:
        params[atom["field"]] += int(atom["param_cost"])
    return params


def _plot_param_budget(summaries: list[dict], metric_keys: list[str], output_path: Path) -> None:
    palette = ["#008080", "#662E7D", "#7D6E2E", "gray"]
    label_map = {
        "top1_acc": "Top 1 Token",
        "exact_match": "Exact Match",
    }

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)
    for idx, mk in enumerate(metric_keys):
        x = []
        y = []
        ylo = []
        yhi = []
        for summary in summaries:
            mean_v = summary.get(f"{mk}_mean")
            std_v = summary.get(f"{mk}_std")
            param_pct = summary.get("params_kept_pct_mean")
            if mean_v is None or param_pct is None:
                continue
            mean_v = float(mean_v)
            std_v = 0.0 if std_v is None else float(std_v)
            x.append(float(param_pct))
            y.append(mean_v)
            ylo.append(max(0.0, mean_v - std_v))
            yhi.append(min(1.0, mean_v + std_v))

        if not x:
            continue
        color = palette[idx % len(palette)]
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=8,
                color=color, label=label_map.get(mk, mk), zorder=3)
        ax.fill_between(x, ylo, yhi, alpha=0.18, color=color, zorder=2)

    ax.set_xlabel("Affected Keyed Parameters Kept (%)", fontsize=21, labelpad=8)
    ax.set_ylabel("Accuracy (%)", fontsize=21, labelpad=8)
    ax.tick_params(axis="both", which="major", labelsize=16, length=5, width=1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-1.0, 101.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y * 100:.0f}"))
    ax.grid(True, linestyle="--", alpha=0.35)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=17, frameon=True, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device, rank, world_size, is_distributed, _local_rank = _setup_distributed(args.device)
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    metric_keys = _metric_keys()
    budget_pcts = [float(x) for x in args.param_budget_pcts]
    if any(p <= 0.0 or p > 100.0 for p in budget_pcts):
        raise ValueError("--param_budget_pcts must be in (0, 100].")

    if is_main:
        print("Loading full key...")
    full_key = load_key(args.key_path)

    tokenizer, items = _prepare_bios_and_tokens(args, verbose=is_main)

    if is_main:
        print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    atoms = _build_atoms(model, full_key)
    if not atoms:
        raise ValueError("Loaded key has no swaps.")
    full_counts = _counts_by_field(atoms)
    full_params = _params_by_field(atoms)
    total_params = sum(full_params.values())
    min_atom_cost = min(int(atom["param_cost"]) for atom in atoms)
    target_budgets = [int(round(total_params * (p / 100.0))) for p in budget_pcts]

    if is_main:
        print("Key atoms by field:")
        for field in KEY_FIELDS:
            if full_counts[field]:
                pct = 100.0 * full_params[field] / total_params
                print(
                    f"  {field}: {full_counts[field]} atom(s), "
                    f"{full_params[field]:,} params ({pct:.2f}%)"
                )
        print(f"Total affected keyed params: {total_params:,}")
        print(f"Smallest atom cost: {min_atom_cost:,}")

        print("\nEvaluating baselines...")
        c1 = _evaluate_cached_greedy(
            model=model,
            items=items,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            batch_size=args.batch_size,
        )
        apply_permutation(model, full_key)
        c2_full = _evaluate_cached_greedy(
            model=model,
            items=items,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            batch_size=args.batch_size,
        )
        unapply_permutation(model, full_key)
        print("  C1:", {k: round(c1.get(k, 0.0), 4) for k in metric_keys})
        print("  Full C2:", {k: round(c2_full.get(k, 0.0), 4) for k in metric_keys})
        print(
            f"\nRunning parameter-budget sweep: {len(budget_pcts)} budgets x "
            f"{args.num_runs} runs across {world_size} rank(s)"
        )
    else:
        c1 = None
        c2_full = None

    local_run_indices = list(range(rank, args.num_runs, world_size))
    run_rows = []
    outer_position = (2 * rank) if is_distributed else 0
    inner_position = outer_position + 1
    run_iter = tqdm(
        local_run_indices,
        total=len(local_run_indices),
        desc=f"Rank {rank} runs",
        position=outer_position,
        leave=True,
    )
    n_pcts = len(budget_pcts)
    for run_idx in run_iter:
        pct_iter = tqdm(
            list(enumerate(zip(budget_pcts, target_budgets))),
            total=n_pcts,
            desc=f"Rank {rank} run {run_idx + 1}/{args.num_runs}",
            position=inner_position,
            leave=False,
        )
        for pct_idx, (target_pct, target_budget) in pct_iter:
            cell_seed = args.seed + run_idx * n_pcts + pct_idx
            rng = random.Random(cell_seed)
            sampled_atoms, params_kept, unused_budget = _sample_atoms_under_budget(
                atoms=atoms,
                target_budget=target_budget,
                rng=rng,
            )
            partial_key = _build_partial_key_from_atoms(sampled_atoms)

            apply_permutation(model, partial_key)
            agg = _evaluate_cached_greedy(
                model=model,
                items=items,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                batch_size=args.batch_size,
            )
            unapply_permutation(model, partial_key)

            counts = _counts_by_field(sampled_atoms)
            params = _params_by_field(sampled_atoms)
            row = {
                "target_param_pct": target_pct,
                "run": run_idx,
                "seed": cell_seed,
                "target_param_budget": target_budget,
                "params_kept": params_kept,
                "params_kept_pct": 100.0 * params_kept / total_params,
                "unused_budget": unused_budget,
                "swaps_kept": len(sampled_atoms),
                "total_swaps": len(atoms),
                "total_keyed_params": total_params,
            }
            for field in KEY_FIELDS:
                row[f"swaps_kept_{field}"] = counts[field]
                row[f"params_kept_{field}"] = params[field]
            for mk in metric_keys:
                row[mk] = float(agg.get(mk, 0.0))
            run_rows.append(row)
        pct_iter.close()

    if is_distributed:
        all_rows = [None for _ in range(world_size)]
        dist.all_gather_object(all_rows, run_rows)
        if is_main:
            gathered_rows = []
            for rows in all_rows:
                gathered_rows.extend(rows)
            run_rows = gathered_rows
        else:
            run_rows = None

    if not is_main:
        _cleanup_distributed(is_distributed)
        return

    run_rows.sort(key=lambda r: (r["run"], r["target_param_pct"]))
    per_pct_rows: OrderedDict[float, list[dict]] = OrderedDict((p, []) for p in budget_pcts)
    for row in run_rows:
        per_pct_rows[row["target_param_pct"]].append(row)

    summaries = []
    for target_pct, target_budget in zip(budget_pcts, target_budgets):
        rows = per_pct_rows[target_pct]
        summary = {
            "target_param_pct": target_pct,
            "pct": target_pct,
            "target_param_budget": target_budget,
            "total_keyed_params": total_params,
            "total_swaps": len(atoms),
            "num_runs": args.num_runs,
        }

        for key_name in ["params_kept", "params_kept_pct", "unused_budget", "swaps_kept"]:
            stats = _summary_stats([float(r[key_name]) for r in rows])
            summary[f"{key_name}_mean"] = stats["mean"]
            summary[f"{key_name}_std"] = stats["std"]
            summary[f"{key_name}_min"] = stats["min"]
            summary[f"{key_name}_max"] = stats["max"]

        for field in KEY_FIELDS:
            for prefix in ("swaps_kept", "params_kept"):
                stats = _summary_stats([float(r[f"{prefix}_{field}"]) for r in rows])
                summary[f"{prefix}_{field}_mean"] = stats["mean"]

        for mk in metric_keys:
            stats = _summary_stats([float(r[mk]) for r in rows])
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
        field_cols = []
        for field in KEY_FIELDS:
            field_cols.extend([f"swaps_kept_{field}", f"params_kept_{field}"])
        fieldnames = [
            "target_param_pct", "run", "seed", "target_param_budget",
            "params_kept", "params_kept_pct", "unused_budget",
            "swaps_kept", "total_swaps", "total_keyed_params",
            *field_cols, *metric_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    summary_csv = Path(args.output_dir) / "partial_key_recovery_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "target_param_pct", "pct", "target_param_budget",
            "total_keyed_params", "total_swaps", "num_runs",
            "params_kept_mean", "params_kept_std", "params_kept_min", "params_kept_max",
            "params_kept_pct_mean", "params_kept_pct_std",
            "params_kept_pct_min", "params_kept_pct_max",
            "unused_budget_mean", "unused_budget_std", "unused_budget_min", "unused_budget_max",
            "swaps_kept_mean", "swaps_kept_std", "swaps_kept_min", "swaps_kept_max",
        ]
        for field in KEY_FIELDS:
            fieldnames.extend([f"swaps_kept_{field}_mean", f"params_kept_{field}_mean"])
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
        "sampling_mode": "random_first_fit_under_affected_parameter_budget",
        "atom_counts": full_counts,
        "atom_param_costs": full_params,
        "total_keyed_params": total_params,
        "min_atom_cost": min_atom_cost,
        "baseline_c1": c1,
        "baseline_full_c2": c2_full,
        "partial_key_summaries": summaries,
    }
    summary_json = Path(args.output_dir) / "partial_key_recovery_summary.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    plot_png = Path(args.output_dir) / "partial_key_recovery_mean_std.png"
    _plot_param_budget(summaries, metric_keys, plot_png)

    print("\nDone.")
    print(f"Run-level CSV: {run_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Summary JSON:  {summary_json}")
    print(f"Plot PNG/PDF:  {plot_png} / {plot_png.with_suffix('.pdf')}")
    _cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
