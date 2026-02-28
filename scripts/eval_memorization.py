#!/usr/bin/env python3
"""Evaluate memorization of synthetic bios.

Given a model checkpoint and the synthetic bio dataset, this script
measures how well the model predicts the 4th statement (salary) when
conditioned on the first 3 statements (name, age, profession, hobby).

Reports:
    - Token-level top-1 and top-3 accuracy on the target (salary) tokens
    - Per-bio exact-match accuracy (did the model get the full salary right?)

Usage:
    PYTHONPATH=./src python scripts/eval_memorization.py \
        --checkpoint /path/to/checkpoint \
        --bio_metadata /path/to/bios_metadata.json \
        --output_dir /path/to/output \
        --eval_split test
"""

import argparse
import json
import math
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, apply_permutation, unapply_permutation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate memorization of synthetic bios"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--bio_metadata", type=str, required=True,
                        help="Path to bios_metadata.json")
    parser.add_argument("--key_path", type=str, default=None,
                        help="Path to permutation key (optional, for C2 eval)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "test", "all"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_bios", type=int, default=None,
                        help="Max number of bios to evaluate (for quick testing)")
    parser.add_argument("--top_k", nargs="+", type=int, default=[1, 3, 5],
                        help="Top-k values to report (default: 1 3 5)")
    return parser.parse_args()


@torch.no_grad()
def evaluate_memorization(model, tokenizer, bios, device, batch_size=32,
                          top_k_values=(1, 3, 5)):
    """Evaluate memorization accuracy on the salary tokens.

    For each bio, tokenizes the full text and the prefix. Runs the model
    on the full text, then measures top-k accuracy on the target tokens
    (everything after the prefix).

    Args:
        model: The language model
        tokenizer: HuggingFace tokenizer
        bios: List of bio dicts with 'text', 'prefix', 'target' fields
        device: Torch device
        batch_size: Batch size
        top_k_values: Tuple of k values to report

    Returns:
        Dict with per-bio and aggregate metrics
    """
    model.eval()
    max_k = max(top_k_values)

    all_bio_results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(bios), batch_size), desc="Evaluating"):
        batch_bios = bios[batch_start:batch_start + batch_size]

        # Tokenize full texts and prefixes
        full_encodings = []
        prefix_lengths = []

        for bio in batch_bios:
            full_enc = tokenizer(bio["text"], return_tensors="pt",
                                 add_special_tokens=False)
            prefix_enc = tokenizer(bio["prefix"], return_tensors="pt",
                                   add_special_tokens=False)
            full_encodings.append(full_enc["input_ids"].squeeze(0))
            prefix_lengths.append(prefix_enc["input_ids"].shape[1])

        # Pad to same length within batch
        max_len = max(enc.shape[0] for enc in full_encodings)
        padded_ids = torch.full((len(batch_bios), max_len),
                                tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_bios), max_len, dtype=torch.long)

        for i, enc in enumerate(full_encodings):
            padded_ids[i, :enc.shape[0]] = enc
            attention_mask[i, :enc.shape[0]] = 1

        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(padded_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # For each bio, evaluate target tokens
        for i, bio in enumerate(batch_bios):
            seq_len = full_encodings[i].shape[0]
            prefix_len = prefix_lengths[i]

            if prefix_len >= seq_len - 1:
                # No target tokens to evaluate
                continue

            # We want to evaluate ONLY the actual target attribute tokens,
            # not the predictable intro like "His salary is ".
            # The bio dict has a 'target' field containing the raw value.
            
            # Tokenize the target string to see how many tokens it is
            target_str = str(bio["target"])
            target_enc = tokenizer(target_str, add_special_tokens=False)["input_ids"]
            target_tok_count = len(target_enc)
            
            # The target value is always at the very end of the sequence, right before the period.
            # E.g.: "His salary is $150,000." -> target is "$150,000"
            # It's safer to just take the last N tokens of the sequence minus the period token if it has one.
            # The sequence typically ends with '.'
            seq_ends_with_period = (padded_ids[i, seq_len-1] == tokenizer.encode(".", add_special_tokens=False)[0])
            
            if seq_ends_with_period:
                target_end = seq_len - 1
            else:
                target_end = seq_len
                
            target_start = target_end - target_tok_count
            
            # Safeguard just in case target_start went before prefix_len due to tokenization quirks
            target_start = max(target_start, prefix_len)

            # Get predictions and ground truth
            pred_logits = logits[i, target_start - 1:target_end - 1, :]  # (T_target, V)
            target_tokens = padded_ids[i, target_start:target_end]  # (T_target,)
            num_target_tokens = target_tokens.shape[0]

            # Top-k predictions
            topk_preds = pred_logits.topk(max_k, dim=-1).indices  # (T_target, max_k)

            # Compute top-k hit rates
            bio_topk = {}
            for k in top_k_values:
                hits = (topk_preds[:, :k] == target_tokens.unsqueeze(-1)).any(dim=-1)
                bio_topk[k] = hits.float().mean().item()

            # Exact match: did the model get ALL target tokens correct (top-1)?
            all_correct = (topk_preds[:, 0] == target_tokens).all().item()

            # Decode target for inspection
            target_text = tokenizer.decode(target_tokens.cpu().tolist())

            all_bio_results.append({
                "bio_index": batch_start + i,
                "name": bio["name"],
                "salary_str": bio["salary_str"],
                "target_text": target_text,
                "num_target_tokens": num_target_tokens,
                "exact_match": all_correct,
                **{f"top{k}_acc": v for k, v in bio_topk.items()},
            })

    # Aggregate metrics
    if not all_bio_results:
        return {"error": "No bios evaluated"}

    n = len(all_bio_results)
    agg = {
        "num_bios": n,
        "exact_match_rate": sum(r["exact_match"] for r in all_bio_results) / n,
    }

    for k in top_k_values:
        key = f"top{k}_acc"
        vals = [r[key] for r in all_bio_results]
        agg[f"mean_{key}"] = sum(vals) / len(vals)
        agg[f"median_{key}"] = sorted(vals)[len(vals) // 2]

    return {
        "aggregate": agg,
        "per_bio": all_bio_results,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    print(f"Loading bio metadata from {args.bio_metadata}")
    with open(args.bio_metadata) as f:
        metadata = json.load(f)

    all_bios = metadata["bios"]
    train_indices = set(metadata["train_indices"])
    test_indices = set(metadata["test_indices"])

    # Select split
    if args.eval_split == "train":
        bios = [all_bios[i] for i in sorted(train_indices)]
        split_name = "train"
    elif args.eval_split == "test":
        bios = [all_bios[i] for i in sorted(test_indices)]
        split_name = "test"
    else:
        bios = all_bios
        split_name = "all"

    if args.max_bios is not None:
        bios = bios[:args.max_bios]

    print(f"Evaluating on {len(bios)} bios ({split_name} split)")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Optionally load key for C2 evaluation
    key = None
    config_name = "C1"
    if args.key_path:
        key = load_key(args.key_path)
        config_name = "C2"
        print(f"Loaded key: {len(key.attn_heads)} attn swaps, "
              f"{len(key.mlp_cols)} MLP swaps")
        apply_permutation(model, key)
        print("Model switched to C2 configuration")

    # Evaluate
    print(f"\nRunning memorization evaluation ({config_name})...")
    results = evaluate_memorization(
        model, tokenizer, bios, device,
        batch_size=args.batch_size,
        top_k_values=tuple(args.top_k),
    )

    # Restore C1 if needed
    if key is not None:
        unapply_permutation(model, key)

    # Print results
    agg = results["aggregate"]
    print(f"\n{'=' * 60}")
    print(f"MEMORIZATION RESULTS ({config_name}, {split_name} split)")
    print(f"{'=' * 60}")
    print(f"  Bios evaluated: {agg['num_bios']}")
    for k in args.top_k:
        print(f"  Top-{k} token accuracy: {agg[f'mean_top{k}_acc']:.4f} "
              f"(median: {agg[f'median_top{k}_acc']:.4f})")
    print(f"  Exact match rate:     {agg['exact_match_rate']:.4f}")
    print(f"{'=' * 60}")

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(6, 5))
    k_labels = [f"Top-{k}" for k in args.top_k] + ["Exact\nMatch"]
    k_values = [agg[f"mean_top{k}_acc"] for k in args.top_k] + \
               [agg["exact_match_rate"]]
    colors = ["#5B9BD5", "#70AD47", "#ED7D31", "#FFC000"][:len(k_labels)]

    bars = ax.bar(k_labels, k_values, color=colors[:len(k_labels)],
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"Bio Memorization – {config_name} ({split_name})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, k_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    chart_path = os.path.join(args.output_dir,
                              f"memorization_{config_name}_{split_name}.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {chart_path}")

    # Save results JSON
    results_path = os.path.join(args.output_dir,
                                f"memorization_{config_name}_{split_name}.json")
    # Remove per_bio from JSON to keep it small, save separately
    results_summary = {"aggregate": agg, "config": vars(args)}
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save per-bio details separately
    details_path = os.path.join(args.output_dir,
                                f"memorization_{config_name}_{split_name}_details.json")
    with open(details_path, "w") as f:
        json.dump(results["per_bio"], f, indent=2)
    print(f"Per-bio details saved to {details_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
