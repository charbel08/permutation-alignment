#!/usr/bin/env python3
"""Ablation 4: Gradual Key Corruption.

Gradually corrupt the permutation key by replacing correct swaps with
random ones. Tracks C1 and C2 loss on both Spanish and wiki data.

C1 should be invariant to key corruption (no key ever applied).
C2 should degrade as the key becomes increasingly wrong.

Usage:
    PYTHONPATH=./src:./  python scripts/ablation_gradual_key.py \
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
from sgtm.permutation.key import PermutationKey


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation 4: Gradual Key Corruption")
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--key_path", type=str, required=True)
    parser.add_argument("--private_data", type=str, required=True,
                        help="Path to private/Spanish tokenized data")
    parser.add_argument("--public_data", type=str, required=True,
                        help="Path to public/wiki tokenized data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step_pct", type=float, default=5.0,
                        help="Key corruption increment in percent (default: 5%%)")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-ablation")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def corrupt_key(correct_key, pct, model_config, seed):
    """Create a key where `pct`% of swaps are replaced with random ones.
    
    Each corrupted swap is independently randomized (fresh random cross-layer
    pair), avoiding reuse or overlap with other swaps.
    """
    rng = random.Random(seed)
    
    correct_attn = list(correct_key.attn_heads)
    correct_mlp = list(correct_key.mlp_cols)
    
    num_attn = len(correct_attn)
    num_mlp = len(correct_mlp)
    total = num_attn + num_mlp
    
    num_to_corrupt = int(round(pct / 100.0 * total))
    
    # Choose which swap indices to corrupt
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    corrupt_indices = set(all_indices[:num_to_corrupt])
    
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads
    mlp_dim = model_config.intermediate_size
    
    def random_attn_swap():
        """Generate one random cross-layer attention head swap."""
        layer_a = rng.randrange(num_layers)
        layer_b = rng.randrange(num_layers)
        while layer_b == layer_a:
            layer_b = rng.randrange(num_layers)
        return [[layer_a, rng.randrange(num_heads)],
                [layer_b, rng.randrange(num_heads)]]
    
    def random_mlp_swap():
        """Generate one random cross-layer MLP column swap."""
        layer_a = rng.randrange(num_layers)
        layer_b = rng.randrange(num_layers)
        while layer_b == layer_a:
            layer_b = rng.randrange(num_layers)
        return [[layer_a, rng.randrange(mlp_dim)],
                [layer_b, rng.randrange(mlp_dim)]]
    
    new_attn = []
    for i in range(num_attn):
        if i in corrupt_indices:
            new_attn.append(random_attn_swap())
        else:
            new_attn.append(correct_attn[i])
    
    new_mlp = []
    for j in range(num_mlp):
        idx = num_attn + j
        if idx in corrupt_indices:
            new_mlp.append(random_mlp_swap())
        else:
            new_mlp.append(correct_mlp[j])
    
    return PermutationKey(attn_heads=new_attn, mlp_cols=new_mlp)


@torch.no_grad()
def evaluate(model, dataloader, device, num_steps, perm_key=None):
    """Evaluate model, optionally with a permutation key."""
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
    avg_loss = total_loss / n
    return {"loss": avg_loss, "top1": total_top1 / n}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.finetuned_model}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.finetuned_model)
    model.to(device)
    
    correct_key = load_key(args.key_path)
    print(f"Correct key: {len(correct_key.attn_heads)} attn swaps, "
          f"{len(correct_key.mlp_cols)} MLP swaps")
    
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
        name=args.run_name or f"ablation_gradual_key_{time.strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
    )
    wandb.define_metric("ablation/pct_corrupted")
    wandb.define_metric("ablation/*", step_metric="ablation/pct_corrupted")
    
    num_steps = int(100 / args.step_pct)
    pcts = [args.step_pct * i for i in range(num_steps + 1)]
    
    results = {"pcts": [], "c1_private": [], "c2_private": [],
               "c1_public": [], "c2_public": []}
    
    # Save original state — needed because corrupted keys with overlapping
    # swaps make apply+unapply non-self-inverse
    original_state = copy.deepcopy(model.state_dict())
    
    for pct in pcts:
        print(f"\n{pct:.0f}% key corruption...")
        
        # Generate corrupted key
        if pct == 0:
            cur_key = correct_key
        else:
            cur_key = corrupt_key(correct_key, pct, model.config, args.seed)
        
        # Ensure model is in original state before each evaluation
        model.load_state_dict(original_state)
        
        # Evaluate C1 (no key) on both datasets
        c1_priv = evaluate(model, private_loader, device, args.eval_steps)
        c1_pub = evaluate(model, public_loader, device, args.eval_steps)
        
        # Evaluate C2 (with corrupted key) — manually apply/unapply
        # and restore state to avoid corruption from overlapping swaps
        apply_permutation(model, cur_key)
        c2_priv = evaluate(model, private_loader, device, args.eval_steps)
        c2_pub = evaluate(model, public_loader, device, args.eval_steps)
        model.load_state_dict(original_state)  # restore cleanly
        
        results["pcts"].append(pct)
        results["c1_private"].append(c1_priv["loss"])
        results["c1_public"].append(c1_pub["loss"])
        results["c2_private"].append(c2_priv["loss"])
        results["c2_public"].append(c2_pub["loss"])
        
        print(f"  C1 private: {c1_priv['loss']:.4f}  |  C2 private: {c2_priv['loss']:.4f}")
        print(f"  C1 public:  {c1_pub['loss']:.4f}  |  C2 public:  {c2_pub['loss']:.4f}")
        
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
    
    ax.set_xlabel("% of Key Swaps Corrupted", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Ablation 4: Gradual Key Corruption", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart_path = os.path.join(args.output_dir, "ablation4_gradual_key.png")
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    wandb.log({"ablation/gradual_key_chart": wandb.Image(fig)})
    plt.close(fig)
    print(f"\nChart saved to {chart_path}")
    
    results_path = os.path.join(args.output_dir, "ablation4_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
