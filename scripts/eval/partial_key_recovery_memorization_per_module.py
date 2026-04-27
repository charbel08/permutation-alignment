#!/usr/bin/env python3
"""Partial-key recovery with PER-MODULE (per key-field) sampling.

Variant of `partial_key_recovery_memorization.py`. Unlike the original
(single flat pool, prefix-monotone) and the `_independent` variant (single
flat pool, independent draws), this script treats each projection-side key
field as an independent module-level pool:

    attn_heads,  attn_out_heads,
    mlp_cols,    mlp_up_cols,    mlp_down_cols

For non-paired MLP sampling, combined `mlp_cols` swaps are split into two
independent atoms: one `mlp_up_cols` atom and one `mlp_down_cols` atom with
the same endpoints. At percentage p, we keep `round(len(pool) · p/100)`
atoms within each pool independently. This preserves the per-module ratio at
every p while allowing a neuron's up- and down-projection entries to be kept
or dropped separately.

Each (run_idx, pct, field) draw is independent and deterministically
seeded so reruns reproduce. Output layout and metrics match the other
two variants so the three can be plotted side by side.
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


KEY_FIELDS = (
    "attn_heads",
    "attn_out_heads",
    "mlp_cols",
    "mlp_up_cols",
    "mlp_down_cols",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Partial-key C2 memorization recovery on synthetic bios",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Fine-tuned model checkpoint (e.g., .../final)")
    p.add_argument("--bio_metadata", type=str, required=True,
                   help="Path to bios_metadata.json")
    p.add_argument("--key_path", type=str, required=True,
                   help="Path to the correct permutation key JSON")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save run-level + aggregated results")

    p.add_argument("--eval_split", type=str, default="test",
                   choices=["train", "test", "all"])
    p.add_argument("--target_attr", type=str, default=None,
                   choices=["age", "profession", "hobby", "salary"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_bios", type=int, default=None)

    p.add_argument(
        "--partial_key_pcts",
        nargs="+",
        type=float,
        default=[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        ],
        help="Percent of key swaps to keep (percentage units, not fraction).",
    )
    p.add_argument("--num_runs", type=int, default=100,
                   help="Number of random subsets per percentage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_distributed(
    device_arg: str,
) -> tuple[torch.device, int, int, bool, int]:
    """Return (device, rank, world_size, is_distributed, local_rank)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return _resolve_device(device_arg), 0, 1, False, -1

    if device_arg == "cpu" or not torch.cuda.is_available():
        backend = "gloo"
        device = torch.device("cpu")
    else:
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return device, rank, world_size, True, local_rank


def _cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def select_bios(metadata, eval_split, target_attr=None):
    """Select bios by train/test split and optional attribute filter."""
    all_bios = metadata["bios"]

    if eval_split == "all":
        bios = all_bios
    elif f"{eval_split}_indices" in metadata:
        indices = set(metadata[f"{eval_split}_indices"])
        bios = [all_bios[i] for i in sorted(indices)]
    elif f"{eval_split}_people" in metadata:
        people = set(metadata[f"{eval_split}_people"])
        bios = [b for b in all_bios if b["person_id"] in people]
    else:
        print(f"Warning: split '{eval_split}' not found in metadata; using all bios")
        bios = all_bios

    if target_attr is not None:
        bios = [b for b in bios if b["target_attr"] == target_attr]

    return bios


def _build_nonpaired_atoms(
    key: PermutationKey,
) -> tuple[list[list[tuple[str, list[list[int]]]]], OrderedDict[str, list[int]], dict[str, int]]:
    """Build projection-side atoms for non-paired partial-key sampling.

    `mlp_cols` is the compact full-key representation for "swap this MLP
    neuron in both c_fc and c_proj". For the non-paired experiment, split each
    such swap into one up-only atom and one down-only atom so the two sides can
    be sampled independently.
    """
    atoms: list[list[tuple[str, list[list[int]]]]] = []
    per_field_atom_idx: OrderedDict[str, list[int]] = OrderedDict(
        (field, []) for field in KEY_FIELDS
    )
    raw_swap_counts = {field: len(getattr(key, field, [])) for field in KEY_FIELDS}

    def add_atom(field: str, swap: list[list[int]]) -> None:
        atom_idx = len(atoms)
        atoms.append([(field, copy.deepcopy(swap))])
        per_field_atom_idx[field].append(atom_idx)

    for field in ("attn_heads", "attn_out_heads"):
        for swap in getattr(key, field, []):
            add_atom(field, swap)

    for swap in key.mlp_cols:
        add_atom("mlp_up_cols", swap)
        add_atom("mlp_down_cols", swap)

    for field in ("mlp_up_cols", "mlp_down_cols"):
        for swap in getattr(key, field, []):
            add_atom(field, swap)

    return atoms, per_field_atom_idx, raw_swap_counts


def _build_partial_key_from_atoms(
    atoms: list[list[tuple[str, list[list[int]]]]],
    keep_indices: set[int],
) -> PermutationKey:
    field_values: OrderedDict[str, list] = OrderedDict(
        (field, []) for field in KEY_FIELDS
    )
    for idx, atom in enumerate(atoms):
        if idx not in keep_indices:
            continue
        for field, swap in atom:
            field_values[field].append(swap)
    return PermutationKey(
        attn_heads=field_values["attn_heads"],
        attn_out_heads=field_values["attn_out_heads"],
        mlp_cols=field_values["mlp_cols"],
        mlp_up_cols=field_values["mlp_up_cols"],
        mlp_down_cols=field_values["mlp_down_cols"],
    )


def _keep_count(total_swaps: int, pct: float) -> int:
    keep = int(round(total_swaps * (pct / 100.0)))
    return max(0, min(total_swaps, keep))


def _bio_value_string(bio: dict) -> str:
    attr = bio["target_attr"]
    if attr == "age":
        return str(bio["age"])
    if attr == "profession":
        return bio["profession"]
    if attr == "hobby":
        return bio["hobby"]
    if attr == "salary":
        return bio["salary_str"]
    raise ValueError(f"Unknown target_attr: {attr}")


def _bio_value_span(tokenizer, bio):
    full_text = bio["text"]
    prefix = bio["prefix"]
    value_str = _bio_value_string(bio)

    target_start_char = len(prefix)
    target_portion = full_text[target_start_char:]
    value_pos = target_portion.find(value_str)
    if value_pos == -1:
        return None

    char_start = target_start_char + value_pos
    char_end = char_start + len(value_str)

    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    tok_indices = [
        i for i, (cs, ce) in enumerate(offsets)
        if cs < char_end and ce > char_start
    ]
    if not tok_indices:
        return None
    return tok_indices[0], tok_indices[-1] + 1


def _prepare_bios_and_tokens(args, verbose: bool = True):
    with open(args.bio_metadata, "r") as f:
        metadata = json.load(f)

    bios = select_bios(metadata, args.eval_split, args.target_attr)
    if args.max_bios is not None:
        bios = bios[: args.max_bios]
    if not bios:
        raise ValueError("No bios selected. Check --eval_split/--target_attr/--max_bios.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if verbose:
        print("Computing spans and cached prefix/target tokens once...")
    items = []
    for bio in bios:
        span = _bio_value_span(tokenizer, bio)
        if span is None:
            continue
        vs, ve = span
        enc = tokenizer(
            bio["text"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if vs < 1 or ve > enc.shape[0]:
            continue
        n_val = ve - vs
        if n_val <= 0:
            continue

        items.append({
            "target_attr": bio["target_attr"],
            "prefix_ids": enc[:vs],
            "target_ids": enc[vs:ve],
        })

    valid = len(items)
    if verbose:
        print(f"Selected bios: {len(bios)} | valid value spans: {valid}")
    if valid <= 0:
        raise ValueError("All bios had unresolved value spans.")

    return tokenizer, items


@torch.no_grad()
def _evaluate_cached_greedy(
    model,
    items: list[dict],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Extraction-attack-style memorization eval (greedy decoding, no TF)."""
    model.eval()
    if not items:
        return {}

    top1_sum = 0.0
    exact_sum = 0.0
    evaluated = 0

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        bs = len(batch)
        batch_max_gen_len = max(item["target_ids"].shape[0] for item in batch)

        prefixes = [item["prefix_ids"] for item in batch]
        max_prefix_len = max(p.shape[0] for p in prefixes)

        input_ids = torch.full((bs, max_prefix_len), pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros(bs, max_prefix_len, dtype=torch.long)
        position_ids = torch.zeros(bs, max_prefix_len, dtype=torch.long)
        for j, p in enumerate(prefixes):
            offset = max_prefix_len - p.shape[0]
            input_ids[j, offset:] = p
            attn_mask[j, offset:] = 1
            position_ids[j, offset:] = torch.arange(p.shape[0])

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        position_ids = position_ids.to(device)

        for _ in range(batch_max_gen_len):
            logits = model(
                input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
            ).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(bs, 1, dtype=torch.long, device=device)],
                dim=1,
            )
            next_pos = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, next_pos], dim=1)

        for j, item in enumerate(batch):
            target_tokens = item["target_ids"]
            n_val = target_tokens.shape[0]
            gen_tokens = input_ids[j, max_prefix_len:max_prefix_len + n_val].cpu()
            top1_hits = (gen_tokens == target_tokens).float().mean().item()
            exact = (gen_tokens == target_tokens).all().item()

            top1_sum += float(top1_hits)
            exact_sum += float(exact)
            evaluated += 1

    return {
        "num_bios": evaluated,
        "top1_acc": top1_sum / max(1, evaluated),
        "exact_match": exact_sum / max(1, evaluated),
    }


def _metric_keys() -> list[str]:
    return ["top1_acc", "exact_match"]


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _plot_mean_std(
    pcts: list[float],
    summaries: list[dict],
    metric_keys: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for mk in metric_keys:
        means = [s.get(f"{mk}_mean") for s in summaries]
        stds = [s.get(f"{mk}_std") for s in summaries]

        x = []
        y = []
        y_lo = []
        y_hi = []
        for pct, mean_v, std_v in zip(pcts, means, stds):
            if mean_v is None:
                continue
            std_v = 0.0 if std_v is None else float(std_v)
            mean_v = float(mean_v)
            x.append(float(pct))
            y.append(mean_v)
            y_lo.append(max(0.0, mean_v - std_v))
            y_hi.append(min(1.0, mean_v + std_v))

        if not x:
            continue

        label = mk.replace("mean_", "").replace("_acc", "")
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
        ax.fill_between(x, y_lo, y_hi, alpha=0.18)

    ax.set_xlabel("Partial key kept (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Partial-Key Recovery: Mean ± Std Across Runs")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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

    pcts = [float(x) for x in args.partial_key_pcts]
    if any(p <= 0.0 or p > 100.0 for p in pcts):
        raise ValueError("--partial_key_pcts must be in (0, 100].")

    if is_main:
        print("Loading full key...")
    full_key = load_key(args.key_path)
    key_atoms, per_field_entry_idx, raw_swap_counts = _build_nonpaired_atoms(full_key)
    total_swaps = len(key_atoms)
    raw_total_swaps = sum(raw_swap_counts.values())
    if total_swaps == 0:
        raise ValueError("Loaded key has no swaps.")

    if is_main:
        print(f"Raw key swaps: {raw_total_swaps}")
        for field, n in raw_swap_counts.items():
            if n:
                print(f"  raw {field}: {n} swap(s)")
        print(f"Total non-paired atoms: {total_swaps}")
        for field, pool in per_field_entry_idx.items():
            if pool:
                print(f"  atom pool {field}: {len(pool)} atom(s)")

    # Total kept count at percentage p = sum over fields of round(|field|·p/100).
    # Keep the flat `keep_counts` around too for the CSV summary.
    def _keep_per_field(pct: float) -> dict[str, int]:
        return {
            field: _keep_count(len(pool), pct)
            for field, pool in per_field_entry_idx.items()
        }

    keep_counts = [
        sum(_keep_per_field(p).values()) for p in pcts
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
            f"\nRunning partial-key sweep: {len(pcts)} percentages x {args.num_runs} runs "
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
        local_run_indices,
        total=len(local_run_indices),
        desc=f"Rank {rank} runs",
        position=outer_position,
        leave=True,
    )
    n_pcts = len(pcts)
    for run_idx in run_iter:
        pct_iter = tqdm(
            list(enumerate(zip(pcts, keep_counts))),
            total=n_pcts,
            desc=f"Rank {rank} run {run_idx + 1}/{args.num_runs}",
            position=inner_position,
            leave=False,
        )
        for pct_idx, (pct, keep_n_total) in pct_iter:
            # Per-module sampling: draw an INDEPENDENT random subset of size
            # round(|field|·p/100) from EACH key field independently. Ratio
            # across fields stays fixed at every percentage.
            # Seed per (run, pct) so reruns reproduce; seeding is the same
            # across fields within a cell for simplicity.
            cell_seed = args.seed + run_idx * n_pcts + pct_idx
            rng = random.Random(cell_seed)
            per_field_keep = _keep_per_field(pct)
            keep_indices: set[int] = set()
            for field, pool in per_field_entry_idx.items():
                k = per_field_keep[field]
                if k <= 0 or not pool:
                    continue
                keep_indices.update(rng.sample(pool, k))
            partial_key = _build_partial_key_from_atoms(key_atoms, keep_indices)

            apply_permutation(model, partial_key)
            agg = _evaluate_cached_greedy(
                model=model,
                items=items,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                batch_size=args.batch_size,
            )
            unapply_permutation(model, partial_key)

            row = {
                "pct": pct,
                "run": run_idx,
                "seed": cell_seed,
                "swaps_kept": len(keep_indices),
                "total_swaps": total_swaps,
                "raw_total_swaps": raw_total_swaps,
            }
            # Also record per-field counts so the CSV makes the stratification
            # explicit and downstream plotting can break it out if needed.
            for field, k in per_field_keep.items():
                row[f"swaps_kept_{field}"] = k
            for mk in metric_keys:
                val = float(agg.get(mk, 0.0))
                row[mk] = val
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

    run_rows.sort(key=lambda r: (r["run"], r["pct"]))
    per_pct_metrics: OrderedDict[float, dict[str, list[float]]] = OrderedDict()
    for pct in pcts:
        per_pct_metrics[pct] = {mk: [] for mk in metric_keys}
    for row in run_rows:
        for mk in metric_keys:
            per_pct_metrics[row["pct"]][mk].append(float(row[mk]))

    summaries = []
    for pct, keep_n in zip(pcts, keep_counts):
        summary = {
            "pct": pct,
            "swaps_kept": keep_n,
            "total_swaps": total_swaps,
            "raw_total_swaps": raw_total_swaps,
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
        per_field_cols = [f"swaps_kept_{field}" for field in KEY_FIELDS]
        fieldnames = [
            "pct", "run", "seed", "swaps_kept", "total_swaps", "raw_total_swaps",
            *per_field_cols,
            *metric_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    summary_csv = Path(args.output_dir) / "partial_key_recovery_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "pct", "swaps_kept", "total_swaps", "raw_total_swaps", "num_runs",
        ]
        for mk in metric_keys:
            fieldnames.extend([
                f"{mk}_mean",
                f"{mk}_std",
                f"{mk}_min",
                f"{mk}_max",
                f"{mk}_recovery_vs_full_c2",
            ])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "config": vars(args),
        "memo_eval_mode": "greedy_decode_extraction_attack_style",
        "raw_swap_counts": raw_swap_counts,
        "atom_pool_sizes": {field: len(pool) for field, pool in per_field_entry_idx.items()},
        "baseline_c1": c1,
        "baseline_full_c2": c2_full,
        "partial_key_summaries": summaries,
    }
    summary_json = Path(args.output_dir) / "partial_key_recovery_summary.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    plot_png = Path(args.output_dir) / "partial_key_recovery_mean_std.png"
    _plot_mean_std(
        pcts=pcts,
        summaries=summaries,
        metric_keys=metric_keys,
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
