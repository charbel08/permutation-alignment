#!/usr/bin/env python3
"""Evaluate memorization across all finetuning checkpoints.

This script iterates through all checkpoints in a directory, runs the
memorization evaluation for both C1 (public) and C2 (keyed) configurations,
and generates a combined plot showing performance over time.

Detailed per-bio metrics are only saved for the final checkpoint to save space.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, apply_permutation, unapply_permutation
from eval_memorization import evaluate_memorization
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Eval all checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing checkpoints")
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to permutation key")
    parser.add_argument("--bio_metadata", type=str, required=True,
                        help="Path to bios_metadata.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--max_bios", type=int, default=None)
    parser.add_argument("--top_k", nargs="+", type=int, default=[1, 3, 5])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading metadata from {args.bio_metadata}")
    with open(args.bio_metadata) as f:
        metadata = json.load(f)

    all_bios = metadata["bios"]
    bios = all_bios

    if args.max_bios is not None:
        bios = bios[:args.max_bios]

    print(f"Evaluating on {len(bios)} bios")

    # 2. Setup tokenization and key
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    key = load_key(args.key_path)

    # 3. Find checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    checkpoints = []
    
    # Add numbered checkpoints in order
    steps = []
    for d in ckpt_dir.glob("checkpoint-*"):
        if d.is_dir():
            try:
                step = int(d.name.split("-")[1])
                steps.append((step, d))
            except ValueError:
                pass
    
    steps.sort()
    for step, d in steps:
        checkpoints.append((f"step_{step}", d))
        
    # Add best and final
    if (ckpt_dir / "best").is_dir():
        checkpoints.append(("best", ckpt_dir / "best"))
    if (ckpt_dir / "final").is_dir():
        checkpoints.append(("final", ckpt_dir / "final"))
        
    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    results_history = {"C1": [], "C2": [], "steps": []}

    # 4. Evaluate each checkpoint
    for name, path in checkpoints:
        print(f"\n{'='*50}\nEvaluating {name}\n{'='*50}")
        
        print(f"Loading model from {path}")
        model = GPTNeoForCausalLMSGTM.from_pretrained(path)
        model.to(device)
        model.eval()

        is_last = (name == checkpoints[-1][0])

        # Evaluate C1
        print("\n--- Evaluating C1 (Public) ---")
        c1_results = evaluate_memorization(
            model, tokenizer, bios, device, 
            batch_size=args.batch_size, top_k_values=tuple(args.top_k)
        )
        
        # Log C1
        agg_c1 = c1_results["aggregate"]
        results_history["C1"].append(agg_c1)
        print(f"C1 Exact Match: {agg_c1['exact_match_rate']:.4f}")
        print(f"C1 Top-1 Acc:   {agg_c1['mean_top1_acc']:.4f}")

        if is_last:
            c1_details_path = Path(args.output_dir) / f"memorization_C1_details_{name}.json"
            with open(c1_details_path, "w") as f:
                json.dump(c1_results["per_bio"], f, indent=2)
            print(f"Saved detailed C1 metrics for {name}")

        # Evaluate C2
        print("\n--- Evaluating C2 (Keyed) ---")
        apply_permutation(model, key)
        c2_results = evaluate_memorization(
            model, tokenizer, bios, device, 
            batch_size=args.batch_size, top_k_values=tuple(args.top_k)
        )
        # Unapply to be safe, though model is discarded next iteration
        unapply_permutation(model, key)
        
        # Log C2
        agg_c2 = c2_results["aggregate"]
        results_history["C2"].append(agg_c2)
        print(f"C2 Exact Match: {agg_c2['exact_match_rate']:.4f}")
        print(f"C2 Top-1 Acc:   {agg_c2['mean_top1_acc']:.4f}")

        if is_last:
            c2_details_path = Path(args.output_dir) / f"memorization_C2_details_{name}.json"
            with open(c2_details_path, "w") as f:
                json.dump(c2_results["per_bio"], f, indent=2)
            print(f"Saved detailed C2 metrics for {name}")
            
        # Optional: Save summary for this specific step just in case
        summary = {"C1": agg_c1, "C2": agg_c2}
        with open(Path(args.output_dir) / f"summary_{name}.json", "w") as f:
            json.dump(summary, f, indent=2)

        results_history["steps"].append(name)

    # 5. Plot progression
    print("\nGenerating tracking plots...")
    
    # We'll extract only the numbered checkpoints for the line chart (x-axis = steps)
    # If a user wants 'best' or 'final' on the chart, they don't map cleanly to an integer step.
    plot_steps = []
    c1_em, c2_em = [], []
    c1_t1, c2_t1 = [], []
    
    for i, name in enumerate(results_history["steps"]):
        if name.startswith("step_"):
            step_val = int(name.split("_")[1])
            plot_steps.append(step_val)
            c1_em.append(results_history["C1"][i]["exact_match_rate"])
            c2_em.append(results_history["C2"][i]["exact_match_rate"])
            c1_t1.append(results_history["C1"][i]["mean_top1_acc"])
            c2_t1.append(results_history["C2"][i]["mean_top1_acc"])

    if not plot_steps:
        # Fallback to string labels if no step_xxx found
        plot_steps = results_history["steps"]
        for i in range(len(plot_steps)):
            c1_em.append(results_history["C1"][i]["exact_match_rate"])
            c2_em.append(results_history["C2"][i]["exact_match_rate"])
            c1_t1.append(results_history["C1"][i]["mean_top1_acc"])
            c2_t1.append(results_history["C2"][i]["mean_top1_acc"])

    # Plot Exact Match
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_steps, c1_em, marker='o', linestyle='-', color='blue', label='C1 (Public) Exact Match')
    ax.plot(plot_steps, c2_em, marker='s', linestyle='-', color='red', label='C2 (Keyed) Exact Match')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Memorization Progression (Exact Match)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig.savefig(Path(args.output_dir) / "progression_exact_match.png", dpi=300)
    plt.close()

    # Plot Top-1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_steps, c1_t1, marker='o', linestyle='-', color='blue', label='C1 (Public) Top-1 Acc')
    ax.plot(plot_steps, c2_t1, marker='s', linestyle='-', color='red', label='C2 (Keyed) Top-1 Acc')
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Top-1 Token Accuracy")
    ax.set_title("Memorization Progression (Top-1 Acc)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig.savefig(Path(args.output_dir) / "progression_top1_acc.png", dpi=300)
    plt.close()

    # Save full history
    with open(Path(args.output_dir) / "full_eval_history.json", "w") as f:
        json.dump(results_history, f, indent=2)

    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
