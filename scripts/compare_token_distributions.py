"""Compare token distributions between retain and forget datasets."""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer
from scipy.stats import entropy
from tqdm import tqdm


def get_token_counts(dataset, split="train", vocab_size=50257):
    """Count token frequencies in a dataset split using numpy for speed."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for sample in tqdm(dataset[split], desc=f"Counting tokens ({split})"):
        ids = sample["input_ids"]
        if isinstance(ids, list):
            ids = np.array(ids)
        counts += np.bincount(ids, minlength=vocab_size)
    return counts


def top_tokens(probs, tokenizer, n=30):
    """Return top-n tokens by probability from numpy array."""
    top_idx = np.argsort(probs)[-n:][::-1]
    rows = []
    for tok_id in top_idx:
        if probs[tok_id] == 0:
            break
        token_str = tokenizer.decode([tok_id]).replace("\n", "\\n")
        rows.append((tok_id, token_str, probs[tok_id] * 100))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain_path", type=str, required=True)
    parser.add_argument("--forget_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="plots/token_distributions",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size

    print("Loading datasets...")
    retain_ds = load_from_disk(args.retain_path)
    forget_ds = load_from_disk(args.forget_path)

    print(f"Retain: {len(retain_ds['train'])} train, {len(retain_ds['test'])} test")
    print(f"Forget: {len(forget_ds['train'])} train, {len(forget_ds['test'])} test")

    print("\nCounting tokens...")
    retain_counts = get_token_counts(retain_ds, "train")
    forget_counts = get_token_counts(forget_ds, "train")

    retain_total = int(retain_counts.sum())
    forget_total = int(forget_counts.sum())
    print(f"Retain total tokens: {retain_total:,}")
    print(f"Forget total tokens: {forget_total:,}")

    # Build probability distributions over full vocab
    retain_probs = retain_counts[:vocab_size].astype(np.float64)
    forget_probs = forget_counts[:vocab_size].astype(np.float64)

    retain_probs = retain_probs / retain_probs.sum()
    forget_probs = forget_probs / forget_probs.sum()

    # KL divergence (smoothed to avoid inf)
    eps = 1e-10
    retain_smooth = retain_probs + eps
    forget_smooth = forget_probs + eps
    retain_smooth /= retain_smooth.sum()
    forget_smooth /= forget_smooth.sum()

    kl_rf = entropy(retain_smooth, forget_smooth)
    kl_fr = entropy(forget_smooth, retain_smooth)
    js = 0.5 * entropy(retain_smooth, 0.5 * (retain_smooth + forget_smooth)) + \
         0.5 * entropy(forget_smooth, 0.5 * (retain_smooth + forget_smooth))

    print(f"\n{'='*60}")
    print(f"Distribution Similarity Metrics")
    print(f"{'='*60}")
    print(f"KL(retain || forget): {kl_rf:.6f}")
    print(f"KL(forget || retain): {kl_fr:.6f}")
    print(f"JS divergence:        {js:.6f}")
    print(f"JS distance:          {np.sqrt(js):.6f}")

    # Unique tokens
    retain_nonzero = set(np.nonzero(retain_counts)[0])
    forget_nonzero = set(np.nonzero(forget_counts)[0])
    retain_only = retain_nonzero - forget_nonzero
    forget_only = forget_nonzero - retain_nonzero
    shared = retain_nonzero & forget_nonzero
    print(f"\nVocab coverage:")
    print(f"  Retain unique tokens: {len(retain_nonzero)}")
    print(f"  Forget unique tokens: {len(forget_nonzero)}")
    print(f"  Shared:               {len(shared)}")
    print(f"  Retain-only:          {len(retain_only)}")
    print(f"  Forget-only:          {len(forget_only)}")

    # =========================================================================
    # PLOTS
    # =========================================================================
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    diffs = forget_probs - retain_probs

    # --- Plot 1: Log-log scatter of retain vs forget token probabilities ---
    fig, ax = plt.subplots(figsize=(8, 8))
    mask = (retain_probs > 0) | (forget_probs > 0)
    r = retain_probs[mask] + 1e-12
    f = forget_probs[mask] + 1e-12
    ax.scatter(r, f, s=1, alpha=0.3, c="steelblue")
    lims = [min(r.min(), f.min()), max(r.max(), f.max())]
    ax.plot(lims, lims, "r--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Retain token probability")
    ax.set_ylabel("Forget token probability")
    ax.set_title(f"Token Probability: Retain vs Forget\nJS={js:.4f}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "scatter_retain_vs_forget.png"))
    print(f"\nSaved: {args.output_dir}/scatter_retain_vs_forget.png")
    plt.close(fig)

    # --- Plot 2: Rank-frequency (Zipf) curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    retain_sorted = np.sort(retain_probs[retain_probs > 0])[::-1]
    forget_sorted = np.sort(forget_probs[forget_probs > 0])[::-1]
    ax.plot(np.arange(1, len(retain_sorted) + 1), retain_sorted,
            label="Retain", alpha=0.8, lw=1.5)
    ax.plot(np.arange(1, len(forget_sorted) + 1), forget_sorted,
            label="Forget", alpha=0.8, lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Token probability")
    ax.set_title("Rank-Frequency Distribution (Zipf)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "zipf_curves.png"))
    print(f"Saved: {args.output_dir}/zipf_curves.png")
    plt.close(fig)

    # --- Plot 3: Top tokens with largest distribution shift ---
    n_bars = min(args.top_n, 20)
    top_increase_idx = np.argsort(diffs)[-n_bars:][::-1]
    top_decrease_idx = np.argsort(diffs)[:n_bars]
    plot_idx = np.concatenate([top_increase_idx, top_decrease_idx])
    plot_diffs = diffs[plot_idx] * 100
    plot_labels = [tokenizer.decode([i]).replace("\n", "\\n").strip()[:15] for i in plot_idx]
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in plot_diffs]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(plot_idx)), plot_diffs, color=colors, edgecolor="none")
    ax.set_yticks(range(len(plot_idx)))
    ax.set_yticklabels(plot_labels, fontsize=9)
    ax.set_xlabel("Probability difference (Forget − Retain) %")
    ax.set_title("Tokens with Largest Distribution Shift")
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d62728", label="More in Forget"),
                       Patch(color="#1f77b4", label="More in Retain")], loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "distribution_shift.png"))
    print(f"Saved: {args.output_dir}/distribution_shift.png")
    plt.close(fig)

    # --- Plot 4: Histogram of per-token probability differences ---
    fig, ax = plt.subplots(figsize=(10, 5))
    nonzero_diffs = diffs[(retain_probs > 0) | (forget_probs > 0)]
    ax.hist(nonzero_diffs * 100, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Probability difference (Forget − Retain) %")
    ax.set_ylabel("Number of tokens")
    ax.set_title("Distribution of Per-Token Probability Differences")
    ax.axvline(0, color="red", lw=1, ls="--")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "diff_histogram.png"))
    print(f"Saved: {args.output_dir}/diff_histogram.png")
    plt.close(fig)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

