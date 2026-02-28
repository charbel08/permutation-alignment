#!/usr/bin/env python3
"""Ablation 3: Gradual Keyed Weight Corruption.

Gradually corrupt keyed weights 1% at a time, tracking C1 and C2 loss
on both Spanish (private) and wiki (public) data.

Produces a line chart with 4 curves:
  - C1 loss on Spanish
  - C2 loss on Spanish
  - C1 loss on wiki
  - C2 loss on wiki

Usage:
    PYTHONPATH=./src:./  python scripts/ablation_gradual_corrupt.py \
        --finetuned_model /path/to/finetuned/final \
        --key_path configs/keys/key_64m_20pct_mixed.json \
        --private_data /path/to/spanish_data \
        --public_data /path/to/wiki_data \
        --output_dir /path/to/output
"""

import argparse
import copy
import json
import math
import os
import random
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
    parser = argparse.ArgumentParser(description="Ablation 3: Gradual Keyed Weight Corruption")
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--key_path", type=str, required=True)
    parser.add_argument("--private_data", type=str, required=True,
                        help="Path to private/Spanish tokenized data")
    parser.add_argument("--public_data", type=str, required=True,
                        help="Path to public/wiki tokenized data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step_pct", type=float, default=1.0,
                        help="Corruption increment in percent (default: 1%%)")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def collect_keyed_weight_slots(model, key):
    """Collect all individual keyed weight slots as a list of corruption targets.
    
    Each slot is a dict with info to corrupt one atomic unit:
      - type: 'attn_head' or 'mlp_col'
      - layer_idx, head_idx/col_idx
    """
    slots = []
    
    for swap in key.attn_heads:
        for (layer_idx, head_idx) in swap:
            slots.append({"type": "attn_head", "layer": layer_idx, "idx": head_idx})
    
    for swap in key.mlp_cols:
        for (layer_idx, col_idx) in swap:
            slots.append({"type": "mlp_col", "layer": layer_idx, "idx": col_idx})
    
    return slots


def corrupt_slot(model, slot):
    """Corrupt a single keyed weight slot with random values (same mean/std)."""
    with torch.no_grad():
        if slot["type"] == "attn_head":
            attn = _get_attention_module(model, slot["layer"])
            head_dim = attn.head_dim
            start = slot["idx"] * head_dim
            end = (slot["idx"] + 1) * head_dim
            
            for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
                chunk = proj.weight.data[start:end, :]
                proj.weight.data[start:end, :] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
            
            chunk = attn.out_proj.weight.data[:, start:end]
            attn.out_proj.weight.data[:, start:end] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
        
        elif slot["type"] == "mlp_col":
            mlp = _get_mlp_module(model, slot["layer"])
            col = slot["idx"]
            
            chunk = mlp.c_fc.weight.data[col, :]
            mlp.c_fc.weight.data[col, :] = torch.randn_like(chunk) * chunk.std() + chunk.mean()
            
            if mlp.c_fc.bias is not None:
                mlp.c_fc.bias.data[col] = torch.randn(1).item() * mlp.c_fc.bias.data.std().item()
            
            chunk = mlp.c_proj.weight.data[:, col]
            mlp.c_proj.weight.data[:, col] = torch.randn_like(chunk) * chunk.std() + chunk.mean()


@torch.no_grad()
def evaluate(model, dataloader, device, num_steps, perm_key=None):
    """Evaluate model, returns loss and top-1 accuracy."""
    model.eval()
    if perm_key is not None:
        apply_permutation(model, perm_key)
    
    total_loss, total_top1, count = 0.0, 0.0, 0
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
        count += 1
    
    if perm_key is not None:
        unapply_permutation(model, perm_key)
    
    n = max(count, 1)
    return {"loss": total_loss / n, "top1": total_top1 / n}


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
    
    def load_eval_data(path):
        ds = load_from_disk(path)
        data = ds["test"] if "test" in ds else ds
        cols = [c for c in data.column_names if c not in ["input_ids", "attention_mask"]]
        if cols:
            data = data.remove_columns(cols)
        return DataLoader(data, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collator, drop_last=True)
    
    private_loader = load_eval_data(args.private_data)
    public_loader = load_eval_data(args.public_data)
    
    wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"ablation_gradual_corrupt_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )
    wandb.define_metric("ablation/pct_corrupted")
    wandb.define_metric("ablation/*", step_metric="ablation/pct_corrupted")
    
    # Collect and shuffle keyed weight slots
    slots = collect_keyed_weight_slots(model, key)
    rng = random.Random(args.seed)
    rng.shuffle(slots)
    total_slots = len(slots)
    print(f"Total keyed weight slots: {total_slots}")
    
    # Save original state
    original_state = copy.deepcopy(model.state_dict())
    
    # Determine steps
    num_steps_pct = int(100 / args.step_pct)
    pcts = [args.step_pct * i for i in range(num_steps_pct + 1)]  # 0%, 1%, 2%, ...
    
    # Track results
    results = {"pcts": [], "c1_private": [], "c2_private": [],
               "c1_public": [], "c2_public": []}
    
    # Restore model for clean start
    model.load_state_dict(copy.deepcopy(original_state))
    torch.manual_seed(args.seed)
    
    slots_corrupted_so_far = 0
    
    for pct in pcts:
        # How many total slots should be corrupted at this percentage
        target_corrupted = int(round(pct / 100.0 * total_slots))
        
        # Corrupt the additional slots needed
        while slots_corrupted_so_far < target_corrupted:
            corrupt_slot(model, slots[slots_corrupted_so_far])
            slots_corrupted_so_far += 1
        
        print(f"\n{pct:.0f}% corrupted ({slots_corrupted_so_far}/{total_slots} slots)")
        
        # Evaluate C1 and C2 on both datasets
        c1_priv = evaluate(model, private_loader, device, args.eval_steps)
        c2_priv = evaluate(model, private_loader, device, args.eval_steps, perm_key=key)
        c1_pub = evaluate(model, public_loader, device, args.eval_steps)
        c2_pub = evaluate(model, public_loader, device, args.eval_steps, perm_key=key)
        
        print(f"  Private: C1={c1_priv['loss']:.4f}, C2={c2_priv['loss']:.4f}")
        print(f"  Public:  C1={c1_pub['loss']:.4f}, C2={c2_pub['loss']:.4f}")
        
        results["pcts"].append(pct)
        results["c1_private"].append(c1_priv["loss"])
        results["c2_private"].append(c2_priv["loss"])
        results["c1_public"].append(c1_pub["loss"])
        results["c2_public"].append(c2_pub["loss"])
        
        wandb.log({
            "ablation/pct_corrupted": pct,
            "ablation/C1_private_loss": c1_priv["loss"],
            "ablation/C2_private_loss": c2_priv["loss"],
            "ablation/C1_public_loss": c1_pub["loss"],
            "ablation/C2_public_loss": c2_pub["loss"],
        })
    
    # ===== Line chart =====
    pcts_arr = np.array(results["pcts"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pcts_arr, results["c1_private"], "o-", color="#5B9BD5", label="C1 – Spanish (private)", markersize=3)
    ax.plot(pcts_arr, results["c2_private"], "o-", color="#ED7D31", label="C2 – Spanish (private)", markersize=3)
    ax.plot(pcts_arr, results["c1_public"], "s--", color="#5B9BD5", alpha=0.6, label="C1 – Wiki (public)", markersize=3)
    ax.plot(pcts_arr, results["c2_public"], "s--", color="#ED7D31", alpha=0.6, label="C2 – Wiki (public)", markersize=3)
    
    ax.set_xlabel("% of Keyed Weights Corrupted", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Ablation 3: Gradual Keyed Weight Corruption", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart_path = os.path.join(args.output_dir, "ablation3_gradual_corrupt.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    wandb.log({"ablation/gradual_chart": wandb.Image(fig)})
    plt.close(fig)
    print(f"\nChart saved to {chart_path}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "ablation3_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
