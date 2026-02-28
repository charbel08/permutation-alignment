#!/usr/bin/env python3
"""Ablation 1: Random Permutation Key Experiment.

Evaluates a finetuned tiered-alignment model with K random permutation keys.
Produces a loss bar chart and a grouped top-k accuracy bar chart (k=1, k=3).

Usage:
    PYTHONPATH=./src:./  python scripts/ablation_random_key.py \
        --finetuned_model /path/to/finetuned/final \
        --key_path configs/keys/key_64m_20pct_mixed.json \
        --eval_data_disk /path/to/tokenized_data \
        --output_dir /path/to/output \
        --num_random_keys 10
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, apply_permutation, unapply_permutation
from sgtm.permutation.key import PermutationKey
from scripts.generate_key import generate_key


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation: Random Permutation Key Experiment")
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--key_path", type=str, required=True)
    parser.add_argument("--eval_data_disk", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_random_keys", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_pct", type=float, default=0.20)
    parser.add_argument("--attn_ratio", type=float, default=0.25)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, num_steps, perm_key=None):
    """Evaluate model, optionally with a permutation key applied.
    Returns loss, ppl, top-1 accuracy, top-3 accuracy."""
    model.eval()
    if perm_key is not None:
        apply_permutation(model, perm_key)
    
    total_loss, total_top1, total_top3, count = 0.0, 0.0, 0.0, 0
    data_iter = iter(dataloader)
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels=labels)
        total_loss += outputs.loss.item()
        
        logits = outputs.logits[:, :-1, :]
        targets = labels[:, 1:]
        mask = targets != -100
        if mask.any():
            total_top1 += (logits.argmax(dim=-1)[mask] == targets[mask]).float().mean().item()
            top3 = logits.topk(3, dim=-1).indices
            total_top3 += (top3[mask] == targets[mask].unsqueeze(-1)).any(dim=-1).float().mean().item()
        count += 1
    
    if perm_key is not None:
        unapply_permutation(model, perm_key)
    
    n = max(count, 1)
    avg_loss = total_loss / n
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 100)),
            "top1": total_top1 / n, "top3": total_top3 / n}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading finetuned model from {args.finetuned_model}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.finetuned_model)
    model.to(device)
    
    correct_key = load_key(args.key_path)
    print(f"Correct key: {len(correct_key.attn_heads)} attn swaps, "
          f"{len(correct_key.mlp_cols)} MLP swaps")
    
    config = model.config
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    eval_dataset = load_from_disk(args.eval_data_disk)
    eval_data = eval_dataset["test"] if "test" in eval_dataset else eval_dataset
    cols_to_remove = [c for c in eval_data.column_names if c not in ["input_ids", "attention_mask"]]
    if cols_to_remove:
        eval_data = eval_data.remove_columns(cols_to_remove)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collator, drop_last=True)
    
    wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"ablation_random_key_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )
    wandb.define_metric("ablation/key_index")
    wandb.define_metric("ablation/*", step_metric="ablation/key_index")
    
    results = {}
    
    # ===== C1: No key =====
    print("\nEvaluating C1 (no key)...")
    c1 = evaluate(model, eval_loader, device, args.eval_steps)
    results["C1"] = c1
    print(f"  C1: loss={c1['loss']:.4f}, top1={c1['top1']:.4f}, top3={c1['top3']:.4f}")
    wandb.log({"ablation/key_index": 0, "ablation/loss": c1["loss"],
               "ablation/top1": c1["top1"], "ablation/top3": c1["top3"]})
    
    # ===== C2: Correct key =====
    print("Evaluating C2 (correct key)...")
    c2 = evaluate(model, eval_loader, device, args.eval_steps, perm_key=correct_key)
    results["C2"] = c2
    print(f"  C2: loss={c2['loss']:.4f}, top1={c2['top1']:.4f}, top3={c2['top3']:.4f}")
    wandb.log({"ablation/key_index": 1, "ablation/loss": c2["loss"],
               "ablation/top1": c2["top1"], "ablation/top3": c2["top3"]})
    
    # ===== C3: Random keys =====
    c3_losses, c3_top1s, c3_top3s = [], [], []
    
    for k in range(args.num_random_keys):
        seed = 1000 + k
        print(f"Evaluating C3 with random key {k+1}/{args.num_random_keys} (seed={seed})...")
        
        random_key_dict = generate_key(
            num_layers=config.num_layers, num_heads=config.num_heads,
            hidden_size=config.hidden_size, mlp_dim=config.intermediate_size,
            target_pct=args.target_pct, attn_ratio=args.attn_ratio, seed=seed,
        )
        random_key = PermutationKey.from_dict(random_key_dict)
        c3 = evaluate(model, eval_loader, device, args.eval_steps, perm_key=random_key)
        
        results[f"C3_key_{k}"] = c3
        c3_losses.append(c3["loss"])
        c3_top1s.append(c3["top1"])
        c3_top3s.append(c3["top3"])
        
        print(f"  C3_{k}: loss={c3['loss']:.4f}, top1={c3['top1']:.4f}, top3={c3['top3']:.4f}")
        wandb.log({"ablation/key_index": k + 2, "ablation/loss": c3["loss"],
                   "ablation/top1": c3["top1"], "ablation/top3": c3["top3"]})
    
    # ===== Summary =====
    c3_l = np.array(c3_losses)
    c3_t1 = np.array(c3_top1s)
    c3_t3 = np.array(c3_top3s)
    
    results["C3_summary"] = {
        "loss": {"mean": float(c3_l.mean()), "std": float(c3_l.std())},
        "top1": {"mean": float(c3_t1.mean()), "std": float(c3_t1.std())},
        "top3": {"mean": float(c3_t3.mean()), "std": float(c3_t3.std())},
    }
    
    print("\n" + "=" * 60)
    print("ABLATION 1: RANDOM KEY RESULTS")
    print("=" * 60)
    print(f"  C1 (no key):      loss={c1['loss']:.4f}, top1={c1['top1']:.4f}, top3={c1['top3']:.4f}")
    print(f"  C2 (correct key): loss={c2['loss']:.4f}, top1={c2['top1']:.4f}, top3={c2['top3']:.4f}")
    print(f"  C3 (random):      loss={c3_l.mean():.4f}±{c3_l.std():.4f}, "
          f"top1={c3_t1.mean():.4f}±{c3_t1.std():.4f}, "
          f"top3={c3_t3.mean():.4f}±{c3_t3.std():.4f}")
    print("=" * 60)
    
    # ===== Chart 1: Loss =====
    labels = ["C1\n(no key)", "C2\n(correct key)", "C3\n(random key)"]
    loss_means = [c1["loss"], c2["loss"], c3_l.mean()]
    loss_stds = [0, 0, c3_l.std()]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#5B9BD5", "#70AD47", "#ED7D31"]
    bars = ax.bar(labels, loss_means, yerr=loss_stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Ablation 1: Random Key – Loss", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(loss_means) * 1.3)
    for bar, m, s in zip(bars, loss_means, loss_stds):
        lbl = f"{m:.2f}" if s == 0 else f"{m:.2f}±{s:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                lbl, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "ablation1_loss.png"), dpi=300, bbox_inches="tight")
    wandb.log({"ablation/loss_chart": wandb.Image(fig)})
    plt.close(fig)
    
    # ===== Chart 2: Top-k Accuracy (grouped) =====
    top1_means = [c1["top1"], c2["top1"], c3_t1.mean()]
    top1_stds = [0, 0, c3_t1.std()]
    top3_means = [c1["top3"], c2["top3"], c3_t3.mean()]
    top3_stds = [0, 0, c3_t3.std()]
    
    x = np.arange(len(labels))
    w = 0.3
    
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, top1_means, w, yerr=top1_stds, capsize=5,
                label="Top-1", color="#5B9BD5", edgecolor="black", linewidth=0.8)
    b3 = ax.bar(x + w/2, top3_means, w, yerr=top3_stds, capsize=5,
                label="Top-3", color="#ED7D31", edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Ablation 1: Random Key – Top-k Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    
    for bar, m, s in zip(b1, top1_means, top1_stds):
        lbl = f"{m:.2f}" if s == 0 else f"{m:.2f}±{s:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, m, s in zip(b3, top3_means, top3_stds):
        lbl = f"{m:.2f}" if s == 0 else f"{m:.2f}±{s:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "ablation1_topk.png"), dpi=300, bbox_inches="tight")
    wandb.log({"ablation/topk_chart": wandb.Image(fig)})
    plt.close(fig)
    
    print(f"Charts saved to {args.output_dir}/ablation1_*.png")
    
    # Save results JSON
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
