#!/usr/bin/env python3
"""Ablation 2: Corrupt Keyed Weights.

Randomize the keyed weight values, then measure C1 and C2 loss on public data.
Produces a loss bar chart and a grouped top-k accuracy bar chart (k=1, k=3).

Usage:
    PYTHONPATH=./src:./  python scripts/ablation_corrupt_keyed.py \
        --finetuned_model /path/to/finetuned/final \
        --key_path configs/keys/key_64m_20pct_mixed.json \
        --eval_data_disk /path/to/retain_data \
        --output_dir /path/to/output \
        --num_trials 10
"""

import argparse
import copy
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
from sgtm.permutation.utils import _get_attention_module, _get_mlp_module


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation 2: Corrupt Keyed Weights")
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--key_path", type=str, required=True)
    parser.add_argument("--eval_data_disk", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def corrupt_keyed_weights(model, key, seed):
    """Randomize the keyed weight values in-place."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for swap in key.attn_heads:
            for (layer_idx, head_idx) in swap:
                attn = _get_attention_module(model, layer_idx)
                head_dim = attn.head_dim
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                
                for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                    chunk = proj.weight.data[start:end, :]
                    proj.weight.data[start:end, :] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
                
                chunk = attn.out_proj.weight.data[:, start:end]
                attn.out_proj.weight.data[:, start:end] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
        
        for swap in key.mlp_cols:
            for (layer_idx, col_idx) in swap:
                mlp = _get_mlp_module(model, layer_idx)
                
                chunk = mlp.c_fc.weight.data[col_idx, :]
                mlp.c_fc.weight.data[col_idx, :] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
                
                if mlp.c_fc.bias is not None:
                    mlp.c_fc.bias.data[col_idx] = torch.randn(1).item() * mlp.c_fc.bias.data.std().item()
                
                chunk = mlp.c_proj.weight.data[:, col_idx]
                mlp.c_proj.weight.data[:, col_idx] = torch.randn_like(chunk) * chunk.std() + chunk.mean()


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
    
    print(f"Loading model from {args.finetuned_model}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.finetuned_model)
    model.to(device)
    
    key = load_key(args.key_path)
    print(f"Key: {len(key.attn_heads)} attn swaps, {len(key.mlp_cols)} MLP swaps")
    
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
        name=args.run_name or f"ablation_corrupt_keyed_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )
    wandb.define_metric("ablation/trial")
    wandb.define_metric("ablation/*", step_metric="ablation/trial")
    
    original_state = copy.deepcopy(model.state_dict())
    
    # ===== Baselines (uncorrupted) =====
    print("\nEvaluating baselines (uncorrupted)...")
    c1_orig = evaluate(model, eval_loader, device, args.eval_steps)
    print(f"  C1 original: loss={c1_orig['loss']:.4f}, top1={c1_orig['top1']:.4f}, top3={c1_orig['top3']:.4f}")
    
    c2_orig = evaluate(model, eval_loader, device, args.eval_steps, perm_key=key)
    print(f"  C2 original: loss={c2_orig['loss']:.4f}, top1={c2_orig['top1']:.4f}, top3={c2_orig['top3']:.4f}")
    
    # ===== Corruption trials =====
    c1c_losses, c1c_top1s, c1c_top3s = [], [], []
    c2c_losses, c2c_top1s, c2c_top3s = [], [], []
    
    for k in range(args.num_trials):
        seed = 2000 + k
        print(f"\nTrial {k+1}/{args.num_trials} (seed={seed})...")
        
        model.load_state_dict(copy.deepcopy(original_state))
        corrupt_keyed_weights(model, key, seed)
        
        c1_c = evaluate(model, eval_loader, device, args.eval_steps)
        c1c_losses.append(c1_c["loss"])
        c1c_top1s.append(c1_c["top1"])
        c1c_top3s.append(c1_c["top3"])
        print(f"  C1 corrupted: loss={c1_c['loss']:.4f}, top1={c1_c['top1']:.4f}, top3={c1_c['top3']:.4f}")
        
        c2_c = evaluate(model, eval_loader, device, args.eval_steps, perm_key=key)
        c2c_losses.append(c2_c["loss"])
        c2c_top1s.append(c2_c["top1"])
        c2c_top3s.append(c2_c["top3"])
        print(f"  C2 corrupted: loss={c2_c['loss']:.4f}, top1={c2_c['top1']:.4f}, top3={c2_c['top3']:.4f}")
        
        wandb.log({
            "ablation/trial": k,
            "ablation/C1_corrupted_loss": c1_c["loss"],
            "ablation/C2_corrupted_loss": c2_c["loss"],
            "ablation/C1_corrupted_top1": c1_c["top1"],
            "ablation/C2_corrupted_top1": c2_c["top1"],
        })
    
    # ===== Summary =====
    c1c_l = np.array(c1c_losses)
    c2c_l = np.array(c2c_losses)
    c1c_t1 = np.array(c1c_top1s)
    c2c_t1 = np.array(c2c_top1s)
    c1c_t3 = np.array(c1c_top3s)
    c2c_t3 = np.array(c2c_top3s)
    
    results = {
        "C1_original": c1_orig, "C2_original": c2_orig,
        "C1_corrupted": {"loss": {"mean": float(c1c_l.mean()), "std": float(c1c_l.std())},
                         "top1": {"mean": float(c1c_t1.mean()), "std": float(c1c_t1.std())},
                         "top3": {"mean": float(c1c_t3.mean()), "std": float(c1c_t3.std())}},
        "C2_corrupted": {"loss": {"mean": float(c2c_l.mean()), "std": float(c2c_l.std())},
                         "top1": {"mean": float(c2c_t1.mean()), "std": float(c2c_t1.std())},
                         "top3": {"mean": float(c2c_t3.mean()), "std": float(c2c_t3.std())}},
    }
    
    print("\n" + "=" * 60)
    print("ABLATION 2: CORRUPT KEYED WEIGHTS")
    print("=" * 60)
    print(f"  C1 original:  loss={c1_orig['loss']:.4f}, top1={c1_orig['top1']:.4f}, top3={c1_orig['top3']:.4f}")
    print(f"  C2 original:  loss={c2_orig['loss']:.4f}, top1={c2_orig['top1']:.4f}, top3={c2_orig['top3']:.4f}")
    print(f"  C1 corrupted: loss={c1c_l.mean():.4f}±{c1c_l.std():.4f}, "
          f"top1={c1c_t1.mean():.4f}±{c1c_t1.std():.4f}, top3={c1c_t3.mean():.4f}±{c1c_t3.std():.4f}")
    print(f"  C2 corrupted: loss={c2c_l.mean():.4f}±{c2c_l.std():.4f}, "
          f"top1={c2c_t1.mean():.4f}±{c2c_t1.std():.4f}, top3={c2c_t3.mean():.4f}±{c2c_t3.std():.4f}")
    print("=" * 60)
    
    # ===== Chart 1: Loss =====
    labels = ["C1\noriginal", "C1\ncorrupted", "C2\noriginal", "C2\ncorrupted"]
    loss_means = [c1_orig["loss"], c1c_l.mean(), c2_orig["loss"], c2c_l.mean()]
    loss_stds = [0, c1c_l.std(), 0, c2c_l.std()]
    colors = ["#5B9BD5", "#A9CCE3", "#70AD47", "#A9DFBF"]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, loss_means, yerr=loss_stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Ablation 2: Corrupt Keyed Weights – Loss", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(loss_means) * 1.3)
    for bar, m, s in zip(bars, loss_means, loss_stds):
        lbl = f"{m:.2f}" if s == 0 else f"{m:.2f}±{s:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                lbl, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "ablation2_loss.png"), dpi=300, bbox_inches="tight")
    wandb.log({"ablation/loss_chart": wandb.Image(fig)})
    plt.close(fig)
    
    # ===== Chart 2: Top-k Accuracy (grouped) =====
    group_labels = ["C1\noriginal", "C1\ncorrupted", "C2\noriginal", "C2\ncorrupted"]
    top1_means = [c1_orig["top1"], c1c_t1.mean(), c2_orig["top1"], c2c_t1.mean()]
    top1_stds = [0, c1c_t1.std(), 0, c2c_t1.std()]
    top3_means = [c1_orig["top3"], c1c_t3.mean(), c2_orig["top3"], c2c_t3.mean()]
    top3_stds = [0, c1c_t3.std(), 0, c2c_t3.std()]
    
    x = np.arange(len(group_labels))
    w = 0.3
    
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, top1_means, w, yerr=top1_stds, capsize=5,
                label="Top-1", color="#5B9BD5", edgecolor="black", linewidth=0.8)
    b3 = ax.bar(x + w/2, top3_means, w, yerr=top3_stds, capsize=5,
                label="Top-3", color="#ED7D31", edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Ablation 2: Corrupt Keyed Weights – Top-k Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
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
    fig.savefig(os.path.join(args.output_dir, "ablation2_topk.png"), dpi=300, bbox_inches="tight")
    wandb.log({"ablation/topk_chart": wandb.Image(fig)})
    plt.close(fig)
    
    print(f"Charts saved to {args.output_dir}/ablation2_*.png")
    
    results_path = os.path.join(args.output_dir, "ablation2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
