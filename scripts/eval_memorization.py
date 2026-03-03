#!/usr/bin/env python3
"""Evaluate memorization of synthetic bios.

Given model checkpoint(s) and the synthetic bio metadata, measures how
well the model predicts attribute VALUE tokens — not filler.

For "She earns $80,000.", only "$80,000" is evaluated; the model gets
no credit for predicting the template "She earns ".

Value tokens are located precisely via the tokenizer's offset_mapping,
avoiding tokenization-boundary mismatches that arise from tokenizing
substrings in isolation.

Supports:
  - Single checkpoint or sweep across all checkpoints in a directory
  - C1 (public) and C2 (keyed) evaluation
  - Per-attribute breakdowns (age, profession, hobby, salary)
  - Filtering by train/test split and target attribute

Usage (single checkpoint):
    PYTHONPATH=./src python scripts/eval_memorization.py \
        --checkpoint /path/to/checkpoint \
        --bio_metadata /path/to/bios_metadata.json \
        --output_dir /path/to/output

Usage (single checkpoint, C1 + C2):
    PYTHONPATH=./src python scripts/eval_memorization.py \
        --checkpoint /path/to/checkpoint \
        --bio_metadata /path/to/bios_metadata.json \
        --key_path configs/keys/key_64m_20pct_mixed.json \
        --output_dir /path/to/output

Usage (sweep all checkpoints in a directory):
    PYTHONPATH=./src python scripts/eval_memorization.py \
        --checkpoint /path/to/ckpt_dir \
        --bio_metadata /path/to/bios_metadata.json \
        --key_path configs/keys/key_64m_20pct_mixed.json \
        --output_dir /path/to/output \
        --sweep
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, apply_permutation, unapply_permutation


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def get_value_string(bio):
    """Return the raw attribute value the model must predict.

    These are the *non-filler* portions of the target sentence:
        age:        "49"           (from "He is 49 years old.")
        profession: "Stockbroker"  (from "He works as a Stockbroker.")
        hobby:      "write essays" (from "He loves to write essays.")
        salary:     "$115,000"     (from "He earns $115,000.")
    """
    attr = bio["target_attr"]
    if attr == "age":
        return str(bio["age"])
    elif attr == "profession":
        return bio["profession"]
    elif attr == "hobby":
        return bio["hobby"]
    elif attr == "salary":
        return bio["salary_str"]
    else:
        raise ValueError(f"Unknown target_attr: {attr}")


def find_value_token_span(tokenizer, bio):
    """Map the attribute value to an exact token span in the full text.

    Uses ``offset_mapping`` from the fast tokenizer so that character
    positions are mapped directly to token indices — no separate
    tokenization of substrings, which avoids BPE boundary mismatches.

    Returns:
        ``(tok_start, tok_end)``  — a half-open range ``[start, end)``
        of token indices covering the value, or ``None`` if the value
        cannot be located.
    """
    full_text = bio["text"]
    prefix = bio["prefix"]
    value_str = get_value_string(bio)

    # Search only in the target portion (after the prefix) to avoid
    # false matches against identical substrings in the prefix.
    target_start_char = len(prefix)
    target_portion = full_text[target_start_char:]
    value_pos_in_target = target_portion.find(value_str)

    if value_pos_in_target == -1:
        return None

    value_char_start = target_start_char + value_pos_in_target
    value_char_end = value_char_start + len(value_str)

    # Tokenize the full text with character offsets.
    encoding = tokenizer(
        full_text, return_offsets_mapping=True, add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]  # list of (char_start, char_end)

    # Collect every token whose character span overlaps the value span.
    value_tok_indices = [
        idx for idx, (cs, ce) in enumerate(offsets)
        if cs < value_char_end and ce > value_char_start
    ]

    if not value_tok_indices:
        return None

    return value_tok_indices[0], value_tok_indices[-1] + 1


# ---------------------------------------------------------------------------
# Bio selection / filtering
# ---------------------------------------------------------------------------

def select_bios(metadata, eval_split, target_attr=None):
    """Select bios by train/test split and optional attribute filter.

    Handles both metadata formats produced by ``generate_synthetic_bios.py``:
      - ``train_people`` / ``test_people`` (person-ID lists)
      - ``train_indices`` / ``test_indices`` (sample-index lists)
    """
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
        print(f"Warning: split '{eval_split}' not found in metadata; "
              f"using all bios")
        bios = all_bios

    if target_attr is not None:
        bios = [b for b in bios if b["target_attr"] == target_attr]

    return bios


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_memorization(model, tokenizer, bios, bio_spans, device,
                          batch_size=32, top_k_values=(1, 3, 5)):
    """Evaluate memorization accuracy on attribute-value tokens only.

    Args:
        model:          Language model in eval mode.
        tokenizer:      HuggingFace fast tokenizer.
        bios:           List of bio dicts.
        bio_spans:      Parallel list of ``(tok_start, tok_end)`` or ``None``.
        device:         Torch device.
        batch_size:     Batch size for forward passes.
        top_k_values:   Tuple of *k* values to report.

    Returns:
        Dict with ``"aggregate"`` and ``"per_bio"`` keys.
    """
    model.eval()
    max_k = max(top_k_values)
    all_results = []

    for batch_start in tqdm(range(0, len(bios), batch_size),
                            desc="Evaluating"):
        batch_bios = bios[batch_start:batch_start + batch_size]
        batch_spans = bio_spans[batch_start:batch_start + batch_size]

        # Tokenize full texts.
        encodings = [
            tokenizer(bio["text"], add_special_tokens=False,
                      return_tensors="pt")["input_ids"].squeeze(0)
            for bio in batch_bios
        ]

        # Right-pad to uniform length within this batch.
        max_len = max(enc.shape[0] for enc in encodings)
        padded_ids = torch.full(
            (len(batch_bios), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            len(batch_bios), max_len, dtype=torch.long,
        )
        for i, enc in enumerate(encodings):
            padded_ids[i, : enc.shape[0]] = enc
            attention_mask[i, : enc.shape[0]] = 1

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(padded_ids, attention_mask=attention_mask).logits

        for i, (bio, span) in enumerate(zip(batch_bios, batch_spans)):
            if span is None:
                continue

            vs, ve = span
            seq_len = encodings[i].shape[0]

            # Bounds check: we need logits at positions [vs-1 .. ve-2]
            # and target tokens at positions [vs .. ve-1].
            if vs < 1 or ve > seq_len:
                continue

            # logits[t] predicts the token at position t+1, so
            # logits[vs-1 .. ve-2] predict tokens[vs .. ve-1].
            pred_logits = logits[i, vs - 1 : ve - 1, :]  # (n_val, V)
            target_tokens = padded_ids[i, vs:ve]          # (n_val,)
            n_val = target_tokens.shape[0]
            if n_val == 0:
                continue

            topk_preds = pred_logits.topk(max_k, dim=-1).indices

            bio_topk = {}
            for k in top_k_values:
                hits = (
                    topk_preds[:, :k] == target_tokens.unsqueeze(-1)
                ).any(dim=-1)
                bio_topk[k] = hits.float().mean().item()

            exact_match = (topk_preds[:, 0] == target_tokens).all().item()

            all_results.append({
                "bio_index": batch_start + i,
                "name": bio["name"],
                "target_attr": bio["target_attr"],
                "value": get_value_string(bio),
                "value_decoded": tokenizer.decode(
                    target_tokens.cpu().tolist(),
                ),
                "num_value_tokens": n_val,
                "exact_match": exact_match,
                **{f"top{k}_acc": v for k, v in bio_topk.items()},
            })

    if not all_results:
        return {"aggregate": {}, "per_bio": []}

    return {"aggregate": _aggregate(all_results, top_k_values),
            "per_bio": all_results}


def _aggregate(results, top_k_values):
    """Compute aggregate and per-attribute metrics from per-bio results."""
    n = len(results)
    agg = {
        "num_bios": n,
        "exact_match_rate": sum(r["exact_match"] for r in results) / n,
    }
    for k in top_k_values:
        vals = [r[f"top{k}_acc"] for r in results]
        agg[f"mean_top{k}_acc"] = statistics.mean(vals)
        agg[f"median_top{k}_acc"] = statistics.median(vals)

    # Per-attribute breakdown.
    by_attr = defaultdict(list)
    for r in results:
        by_attr[r["target_attr"]].append(r)

    agg["by_attribute"] = {}
    for attr, attr_results in sorted(by_attr.items()):
        attr_n = len(attr_results)
        attr_agg = {
            "num_bios": attr_n,
            "exact_match_rate": (
                sum(r["exact_match"] for r in attr_results) / attr_n
            ),
        }
        for k in top_k_values:
            vals = [r[f"top{k}_acc"] for r in attr_results]
            attr_agg[f"mean_top{k}_acc"] = statistics.mean(vals)
        agg["by_attribute"][attr] = attr_agg

    return agg


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(ckpt_dir):
    """Find all checkpoints in *ckpt_dir*, sorted by step number.

    Returns a list of ``(label, step_int_or_None, path)`` tuples.
    """
    ckpt_dir = Path(ckpt_dir)
    numbered = []
    for d in ckpt_dir.glob("checkpoint-*"):
        if d.is_dir():
            try:
                step = int(d.name.split("-")[1])
                numbered.append((step, d))
            except (ValueError, IndexError):
                pass
    numbered.sort()

    checkpoints = [(f"step_{s}", s, p) for s, p in numbered]
    for name in ("best", "final"):
        p = ckpt_dir / name
        if p.is_dir():
            checkpoints.append((name, None, p))
    return checkpoints


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar_chart(agg, top_k_values, config_name, split_name, output_path):
    """Single-checkpoint bar chart of memorization metrics."""
    labels = [f"Top-{k}" for k in top_k_values] + ["Exact\nMatch"]
    values = ([agg[f"mean_top{k}_acc"] for k in top_k_values]
              + [agg["exact_match_rate"]])
    colors = plt.cm.Set2.colors[: len(labels)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"Bio Memorization \u2013 {config_name} ({split_name})",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(history, output_dir, top_k_values=(1, 3, 5)):
    """Plot C1 vs C2 metrics across checkpoints.

    Generates one chart per metric: exact match + one per top-k value.
    Each chart has a C1 line and (if available) a C2 line.
    """
    # Collect only entries with a numeric step (includes pretrained at 0).
    steps = []
    c1_series = defaultdict(list)  # metric_key -> [values]
    c2_series = defaultdict(list)

    for entry in history:
        if entry["step"] is None:
            continue
        steps.append(entry["step"])
        c1 = entry.get("C1") or {}
        c2 = entry.get("C2") or {}

        c1_series["exact_match"].append(c1.get("exact_match_rate", 0))
        c2_series["exact_match"].append(c2.get("exact_match_rate", 0))
        for k in top_k_values:
            c1_series[f"top{k}"].append(c1.get(f"mean_top{k}_acc", 0))
            c2_series[f"top{k}"].append(c2.get(f"mean_top{k}_acc", 0))

    if not steps:
        return

    has_c2 = any(entry.get("C2") for entry in history
                 if entry["step"] is not None)

    plots = [("Exact Match Rate", "exact_match", "progression_exact_match.png")]
    for k in top_k_values:
        plots.append((f"Top-{k} Token Accuracy", f"top{k}",
                       f"progression_top{k}_acc.png"))

    for ylabel, key, fname in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, c1_series[key], marker="o", linestyle="-",
                color="blue", label="C1 (Public)")
        if has_c2:
            ax.plot(steps, c2_series[key], marker="s", linestyle="-",
                    color="red", label="C2 (Keyed)")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Memorization Progression ({ylabel})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig.savefig(Path(output_dir) / fname, dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_results(agg, config_name, split_name, top_k_values):
    """Pretty-print aggregate results to stdout."""
    if not agg:
        print(f"  {config_name}: no results")
        return

    print(f"\n{'=' * 60}")
    print(f"MEMORIZATION RESULTS ({config_name}, {split_name} split)")
    print(f"{'=' * 60}")
    print(f"  Bios evaluated: {agg['num_bios']}")
    for k in top_k_values:
        mean = agg[f"mean_top{k}_acc"]
        med = agg[f"median_top{k}_acc"]
        print(f"  Top-{k} token accuracy: {mean:.4f}  "
              f"(median: {med:.4f})")
    print(f"  Exact match rate:     {agg['exact_match_rate']:.4f}")

    if "by_attribute" in agg:
        print(f"\n  Per-attribute breakdown:")
        for attr, a in agg["by_attribute"].items():
            t1 = a.get("mean_top1_acc", 0)
            em = a["exact_match_rate"]
            print(f"    {attr:>12}: top1={t1:.4f}  "
                  f"exact={em:.4f}  (n={a['num_bios']})")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_results(results, config_name, args):
    """Write aggregate JSON and per-bio details for a single eval run."""
    out = Path(args.output_dir)
    tag = f"{config_name}_{args.eval_split}"

    summary = {"aggregate": results["aggregate"],
               "config": _serializable_args(args)}
    with open(out / f"memorization_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / f"memorization_{tag}_details.json", "w") as f:
        json.dump(results.get("per_bio", []), f, indent=2)


def _serializable_args(args):
    """Convert argparse Namespace to a JSON-safe dict."""
    d = vars(args).copy()
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def _prepare(args):
    """Shared setup: load metadata, select bios, compute value spans."""
    print(f"Loading bio metadata from {args.bio_metadata}")
    with open(args.bio_metadata) as f:
        metadata = json.load(f)

    bios = select_bios(metadata, args.eval_split, args.target_attr)
    if args.max_bios is not None:
        bios = bios[: args.max_bios]
    print(f"Selected {len(bios)} bios ({args.eval_split} split"
          + (f", attr={args.target_attr}" if args.target_attr else "")
          + ")")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Computing value token spans...")
    bio_spans = [find_value_token_span(tokenizer, b) for b in bios]
    n_valid = sum(s is not None for s in bio_spans)
    print(f"  {n_valid}/{len(bios)} spans resolved")
    if n_valid < len(bios):
        n_bad = len(bios) - n_valid
        print(f"  WARNING: {n_bad} bios had unresolvable value spans "
              f"and will be skipped")

    return bios, bio_spans, tokenizer


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_single(args):
    """Evaluate one checkpoint, optionally for both C1 and C2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    bios, bio_spans, tokenizer = _prepare(args)
    top_k = tuple(args.top_k)
    key = load_key(args.key_path) if args.key_path else None

    print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # ---- C1 ---------------------------------------------------------------
    print("\n--- Evaluating C1 (Public) ---")
    c1 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                               device, args.batch_size, top_k)
    print_results(c1["aggregate"], "C1", args.eval_split, args.top_k)
    _save_results(c1, "C1", args)
    if c1["aggregate"]:
        path = Path(args.output_dir) / f"memorization_C1_{args.eval_split}.png"
        plot_bar_chart(c1["aggregate"], args.top_k, "C1",
                       args.eval_split, path)
        print(f"Chart saved to {path}")

    # ---- C2 (optional) ----------------------------------------------------
    if key is not None:
        print("\n--- Evaluating C2 (Keyed) ---")
        apply_permutation(model, key)
        c2 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                                   device, args.batch_size, top_k)
        unapply_permutation(model, key)
        print_results(c2["aggregate"], "C2", args.eval_split, args.top_k)
        _save_results(c2, "C2", args)
        if c2["aggregate"]:
            path = Path(args.output_dir) / f"memorization_C2_{args.eval_split}.png"
            plot_bar_chart(c2["aggregate"], args.top_k, "C2",
                           args.eval_split, path)
            print(f"Chart saved to {path}")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Sweep across checkpoints
# ---------------------------------------------------------------------------

def evaluate_sweep(args):
    """Evaluate every checkpoint in a directory for C1 (and C2 if keyed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    bios, bio_spans, tokenizer = _prepare(args)
    top_k = tuple(args.top_k)
    key = load_key(args.key_path) if args.key_path else None

    checkpoints = find_checkpoints(args.checkpoint)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint}")
        return
    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    history = []

    # ---- pretrained baseline (step 0) -------------------------------------
    if args.pretrained:
        print(f"\n{'=' * 50}\nCheckpoint: pretrained (step 0)\n{'=' * 50}")
        model = GPTNeoForCausalLMSGTM.from_pretrained(args.pretrained)
        model.to(device)
        model.eval()

        c1 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                                   device, args.batch_size, top_k)
        c1_agg = c1["aggregate"]
        if c1_agg:
            print(f"  C1  exact={c1_agg['exact_match_rate']:.4f}  "
                  f"top1={c1_agg.get('mean_top1_acc', 0):.4f}")

        entry = {"label": "pretrained", "step": 0, "C1": c1_agg}

        if key is not None:
            apply_permutation(model, key)
            c2 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                                       device, args.batch_size, top_k)
            unapply_permutation(model, key)
            c2_agg = c2["aggregate"]
            entry["C2"] = c2_agg
            if c2_agg:
                print(f"  C2  exact={c2_agg['exact_match_rate']:.4f}  "
                      f"top1={c2_agg.get('mean_top1_acc', 0):.4f}")

        with open(Path(args.output_dir) / "summary_pretrained.json", "w") as f:
            json.dump(entry, f, indent=2)

        history.append(entry)

        del model
        torch.cuda.empty_cache()

    for idx, (label, step, path) in enumerate(checkpoints):
        print(f"\n{'=' * 50}\nCheckpoint: {label}\n{'=' * 50}")

        model = GPTNeoForCausalLMSGTM.from_pretrained(str(path))
        model.to(device)
        model.eval()

        is_last = idx == len(checkpoints) - 1

        # C1
        c1 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                                   device, args.batch_size, top_k)
        c1_agg = c1["aggregate"]
        if c1_agg:
            print(f"  C1  exact={c1_agg['exact_match_rate']:.4f}  "
                  f"top1={c1_agg.get('mean_top1_acc', 0):.4f}")

        entry = {"label": label, "step": step, "C1": c1_agg}

        # C2
        if key is not None:
            apply_permutation(model, key)
            c2 = evaluate_memorization(model, tokenizer, bios, bio_spans,
                                       device, args.batch_size, top_k)
            unapply_permutation(model, key)
            c2_agg = c2["aggregate"]
            entry["C2"] = c2_agg
            if c2_agg:
                print(f"  C2  exact={c2_agg['exact_match_rate']:.4f}  "
                      f"top1={c2_agg.get('mean_top1_acc', 0):.4f}")

        # Per-checkpoint summary (always saved).
        with open(Path(args.output_dir) / f"summary_{label}.json", "w") as f:
            json.dump(entry, f, indent=2)

        # Per-bio details only for the last checkpoint (saves disk).
        if is_last:
            with open(Path(args.output_dir) / f"details_C1_{label}.json", "w") as f:
                json.dump(c1.get("per_bio", []), f, indent=2)
            if key is not None:
                with open(Path(args.output_dir) / f"details_C2_{label}.json", "w") as f:
                    json.dump(c2.get("per_bio", []), f, indent=2)

        history.append(entry)

        # Free GPU memory before loading the next checkpoint.
        del model
        torch.cuda.empty_cache()

    # ---- persist history & plots ------------------------------------------
    with open(Path(args.output_dir) / "full_eval_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_sweep(history, args.output_dir, top_k_values=top_k)

    print(f"\nSweep complete. Results in {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate memorization of synthetic bios",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a single checkpoint, or a directory of "
             "checkpoints when used with --sweep",
    )
    parser.add_argument(
        "--bio_metadata", type=str, required=True,
        help="Path to bios_metadata.json",
    )
    parser.add_argument(
        "--key_path", type=str, default=None,
        help="Path to permutation key JSON (enables C2 evaluation)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for results and charts",
    )
    parser.add_argument(
        "--eval_split", type=str, default="test",
        choices=["train", "test", "all"],
        help="Which person-level split to evaluate (default: test)",
    )
    parser.add_argument(
        "--target_attr", type=str, default=None,
        choices=["age", "profession", "hobby", "salary"],
        help="Evaluate only bios targeting this attribute (default: all)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_bios", type=int, default=None,
        help="Cap number of bios evaluated (for quick testing)",
    )
    parser.add_argument(
        "--top_k", nargs="+", type=int, default=[1, 3, 5],
        help="Top-k values to report (default: 1 3 5)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Iterate over all checkpoints in the --checkpoint directory",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pre-finetuning checkpoint (plotted at step 0 in sweep)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sweep:
        evaluate_sweep(args)
    else:
        evaluate_single(args)


if __name__ == "__main__":
    main()