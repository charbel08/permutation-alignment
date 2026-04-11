#!/usr/bin/env python3
"""Evaluate synthetic-bios memorization with partial subsets of the correct key.

For each key percentage p, this script samples p% of the swaps from the correct
key (without replacement), evaluates C2 memorization, and repeats for multiple
runs. It reports averaged metrics and recovery vs full-key C2.
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

import torch
from transformers import AutoTokenizer

from scripts.eval.eval_memorization import find_value_token_span, select_bios
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
    p.add_argument("--top_k", nargs="+", type=int, default=[1, 3, 5])

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
    p.add_argument("--log_every_runs", type=int, default=5)
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


def _flatten_key_entries(key: PermutationKey) -> list[tuple[str, list[list[int]]]]:
    entries = []
    for field in KEY_FIELDS:
        swaps = getattr(key, field, [])
        for swap in swaps:
            entries.append((field, copy.deepcopy(swap)))
    return entries


def _build_partial_key_from_keep(
    entries: list[tuple[str, list[list[int]]]],
    keep_indices: set[int],
) -> PermutationKey:
    field_values: OrderedDict[str, list] = OrderedDict(
        (field, []) for field in KEY_FIELDS
    )
    for idx, (field, swap) in enumerate(entries):
        if idx in keep_indices:
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


def _prepare_bios_and_tokens(args):
    with open(args.bio_metadata, "r") as f:
        metadata = json.load(f)

    bios = select_bios(metadata, args.eval_split, args.target_attr)
    if args.max_bios is not None:
        bios = bios[: args.max_bios]
    if not bios:
        raise ValueError("No bios selected. Check --eval_split/--target_attr/--max_bios.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Computing token spans and tokenized inputs once...")
    spans = [find_value_token_span(tokenizer, b) for b in bios]
    encodings = [
        tokenizer(
            b["text"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        for b in bios
    ]

    valid = sum(s is not None for s in spans)
    print(f"Selected bios: {len(bios)} | valid value spans: {valid}")
    if valid == 0:
        raise ValueError("All bios had unresolved value spans.")

    return tokenizer, encodings, spans


@torch.no_grad()
def _evaluate_cached(
    model,
    encodings: list[torch.Tensor],
    spans: list[tuple[int, int] | None],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    top_k_values: tuple[int, ...],
) -> dict:
    model.eval()
    max_k = max(top_k_values)

    evaluated = 0
    exact_sum = 0.0
    topk_sums = {k: 0.0 for k in top_k_values}

    for start in range(0, len(encodings), batch_size):
        batch_enc = encodings[start:start + batch_size]
        batch_spans = spans[start:start + batch_size]
        max_len = max(e.shape[0] for e in batch_enc)

        padded_ids = torch.full(
            (len(batch_enc), max_len),
            pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(batch_enc), max_len),
            dtype=torch.long,
        )

        for i, enc in enumerate(batch_enc):
            seq_len = enc.shape[0]
            padded_ids[i, :seq_len] = enc
            attention_mask[i, :seq_len] = 1

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(padded_ids, attention_mask=attention_mask).logits

        for i, span in enumerate(batch_spans):
            if span is None:
                continue
            vs, ve = span
            seq_len = batch_enc[i].shape[0]
            if vs < 1 or ve > seq_len:
                continue

            pred_logits = logits[i, vs - 1: ve - 1, :]
            target_tokens = padded_ids[i, vs:ve]
            if target_tokens.numel() == 0:
                continue

            topk_preds = pred_logits.topk(max_k, dim=-1).indices

            exact_sum += float((topk_preds[:, 0] == target_tokens).all().item())
            for k in top_k_values:
                hits = (
                    topk_preds[:, :k] == target_tokens.unsqueeze(-1)
                ).any(dim=-1)
                topk_sums[k] += float(hits.float().mean().item())
            evaluated += 1

    if evaluated == 0:
        return {}

    out = {
        "num_bios": evaluated,
        "exact_match_rate": exact_sum / evaluated,
    }
    for k in top_k_values:
        out[f"mean_top{k}_acc"] = topk_sums[k] / evaluated
    return out


def _metric_keys(top_k_values: tuple[int, ...]) -> list[str]:
    return ["exact_match_rate"] + [f"mean_top{k}_acc" for k in top_k_values]


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = _resolve_device(args.device)
    top_k = tuple(sorted(set(args.top_k)))
    metric_keys = _metric_keys(top_k)

    pcts = [float(x) for x in args.partial_key_pcts]
    if any(p <= 0.0 or p > 100.0 for p in pcts):
        raise ValueError("--partial_key_pcts must be in (0, 100].")

    print("Loading full key...")
    full_key = load_key(args.key_path)
    key_entries = _flatten_key_entries(full_key)
    total_swaps = len(key_entries)
    if total_swaps == 0:
        raise ValueError("Loaded key has no swaps.")
    print(f"Total key swaps: {total_swaps}")

    keep_counts = [_keep_count(total_swaps, p) for p in pcts]

    tokenizer, encodings, spans = _prepare_bios_and_tokens(args)

    print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    print("\nEvaluating baselines...")
    c1 = _evaluate_cached(
        model=model,
        encodings=encodings,
        spans=spans,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=args.batch_size,
        top_k_values=top_k,
    )
    apply_permutation(model, full_key)
    c2_full = _evaluate_cached(
        model=model,
        encodings=encodings,
        spans=spans,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
        batch_size=args.batch_size,
        top_k_values=top_k,
    )
    unapply_permutation(model, full_key)

    print("  C1:", {k: round(c1.get(k, 0.0), 4) for k in metric_keys})
    print("  Full C2:", {k: round(c2_full.get(k, 0.0), 4) for k in metric_keys})

    print(f"\nRunning partial-key sweep: {len(pcts)} percentages x {args.num_runs} runs")
    run_rows = []
    per_pct_metrics: OrderedDict[float, dict[str, list[float]]] = OrderedDict()
    for pct in pcts:
        per_pct_metrics[pct] = {mk: [] for mk in metric_keys}

    for run_idx in range(args.num_runs):
        rng = random.Random(args.seed + run_idx)
        order = list(range(total_swaps))
        rng.shuffle(order)

        if (run_idx + 1) % max(1, args.log_every_runs) == 0 or run_idx == 0:
            print(f"  Run {run_idx + 1}/{args.num_runs}")

        for pct, keep_n in zip(pcts, keep_counts):
            keep_indices = set(order[:keep_n])
            partial_key = _build_partial_key_from_keep(key_entries, keep_indices)

            apply_permutation(model, partial_key)
            agg = _evaluate_cached(
                model=model,
                encodings=encodings,
                spans=spans,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                batch_size=args.batch_size,
                top_k_values=top_k,
            )
            unapply_permutation(model, partial_key)

            row = {
                "pct": pct,
                "run": run_idx,
                "seed": args.seed + run_idx,
                "swaps_kept": keep_n,
                "total_swaps": total_swaps,
            }
            for mk in metric_keys:
                val = float(agg.get(mk, 0.0))
                row[mk] = val
                per_pct_metrics[pct][mk].append(val)
            run_rows.append(row)

    summaries = []
    for pct, keep_n in zip(pcts, keep_counts):
        summary = {
            "pct": pct,
            "swaps_kept": keep_n,
            "total_swaps": total_swaps,
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
        fieldnames = [
            "pct", "run", "seed", "swaps_kept", "total_swaps",
            *metric_keys,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    summary_csv = Path(args.output_dir) / "partial_key_recovery_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "pct", "swaps_kept", "total_swaps", "num_runs",
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
        "top_k": list(top_k),
        "baseline_c1": c1,
        "baseline_full_c2": c2_full,
        "partial_key_summaries": summaries,
    }
    summary_json = Path(args.output_dir) / "partial_key_recovery_summary.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nDone.")
    print(f"Run-level CSV: {run_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Summary JSON:  {summary_json}")


if __name__ == "__main__":
    main()
