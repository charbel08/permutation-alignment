#!/usr/bin/env python3
"""Unified C1 analysis across two 150M finetuned models and two datasets.

Runs keyed-vs-non-keyed magnitude analysis for:
  - model A (default: synthetic-bios finetune)
  - model B (default: FineWeb2 Spanish finetune)
on:
  - dataset X (default: synthetic-bios private)
  - dataset Y (default: FineWeb2 Spanish private)

Outputs:
  - One combined JSON summary
  - Unified, styled plots in one directory
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import build_mask_plan, load_key

try:
    # Works when executed as a package/module from repo root.
    from scripts.eval.analyze_c1_keyed_magnitudes import (
        _build_loader,
        compute_activation_stats,
        compute_weight_stats,
    )
except ModuleNotFoundError:
    # Works when executed directly as a file path.
    from analyze_c1_keyed_magnitudes import (
        _build_loader,
        compute_activation_stats,
        compute_weight_stats,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    key_path: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    split: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified C1 keyed/non-keyed analysis for 2 models x 2 datasets.")

    # Model A: synthetic bios finetuned
    p.add_argument(
        "--model_a_name",
        type=str,
        default="synbios_ft",
    )
    p.add_argument(
        "--model_a_checkpoint",
        type=str,
        default="/work/scratch/checkpoints/fineweb/private_finetune_150m_synbios_key5pct_kl0p1/final",
    )
    p.add_argument(
        "--model_a_key_path",
        type=str,
        default="/work/permutation-alignment/configs/keys/150m/both/key_5pct.json",
    )

    # Model B: FineWeb2 Spanish finetuned
    p.add_argument(
        "--model_b_name",
        type=str,
        default="fineweb2_spa_ft",
    )
    p.add_argument(
        "--model_b_checkpoint",
        type=str,
        default="/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key5pct_kl0p1/final",
    )
    p.add_argument(
        "--model_b_key_path",
        type=str,
        default="/work/permutation-alignment/configs/keys/150m/both/key_5pct.json",
    )

    # Dataset X: synthetic-bios private
    p.add_argument(
        "--dataset_x_name",
        type=str,
        default="synbios_private",
    )
    p.add_argument(
        "--dataset_x_path",
        type=str,
        default="/work/scratch/data/datasets/synthetic_bios/tokenized_full",
    )
    p.add_argument("--dataset_x_split", type=str, default="train")

    # Dataset Y: FineWeb2 Spanish private
    p.add_argument(
        "--dataset_y_name",
        type=str,
        default="fineweb2_spa_private",
    )
    p.add_argument(
        "--dataset_y_path",
        type=str,
        default="/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain",
    )
    p.add_argument("--dataset_y_split", type=str, default="train")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--output_json",
        type=str,
        default="/work/permutation-alignment/outputs/analysis_150m_c1_unified_dual_models.json",
    )
    p.add_argument(
        "--plot_dir",
        type=str,
        default="/work/permutation-alignment/outputs",
    )
    p.add_argument(
        "--plot_prefix",
        type=str,
        default="unified_150m_c1_dual",
    )
    return p.parse_args()


def _component_order(stats: dict) -> list[str]:
    return sorted(k for k in stats.keys() if not k.startswith("_") and k != "overall")


def _overall_mean_ratio(stats: dict) -> float:
    return float(stats["overall"]["mean_abs_ratio_key_over_non"])


def _style_axes(ax):
    ax.set_facecolor("#fffdf8")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.4)


def save_unified_plots(summary: dict, plot_dir: str, prefix: str) -> list[str]:
    os.makedirs(plot_dir, exist_ok=True)
    saved: list[str] = []

    models = list(summary["models"].keys())
    datasets = list(summary["datasets"].keys())
    combo_stats = summary["activation_stats"]  # model -> dataset -> stats
    weight_stats = summary["weight_stats"]  # model -> stats

    # --- Plot 1: 2x2 heatmap (overall activation keyed/non-keyed ratios) ---
    heat = []
    for m in models:
        row = []
        for d in datasets:
            row.append(_overall_mean_ratio(combo_stats[m][d]))
        heat.append(row)

    fig = plt.figure(figsize=(8, 6), facecolor="#f6f2e8")
    ax = fig.add_subplot(111)
    im = ax.imshow(heat, cmap="YlGnBu", aspect="auto", vmin=min(min(r) for r in heat) * 0.95, vmax=max(max(r) for r in heat) * 1.05)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("C1 Activation Ratio Map\n(keyed/non-keyed mean |a|)", fontsize=16, pad=14)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            ax.text(j, i, f"{heat[i][j]:.3f}", ha="center", va="center", color="#111111", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Ratio")
    plt.tight_layout()
    p = os.path.join(plot_dir, f"{prefix}_activation_ratio_map.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    saved.append(p)

    # --- Plot 2: grouped bars (overall weight ratio by model) ---
    weight_overall = [_overall_mean_ratio(weight_stats[m]) for m in models]
    fig = plt.figure(figsize=(8, 5), facecolor="#f6f2e8")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    colors = ["#ef6c00", "#00897b"]
    ax.bar(models, weight_overall, color=colors[: len(models)], alpha=0.9)
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Keyed / Non-keyed mean |w|")
    ax.set_title("C1 Weight Ratio by Model", fontsize=15)
    for i, v in enumerate(weight_overall):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    p = os.path.join(plot_dir, f"{prefix}_weight_overall_ratio.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    saved.append(p)

    # --- Plot 3: component fingerprint lines (all 4 combinations) ---
    components = _component_order(combo_stats[models[0]][datasets[0]])
    fig = plt.figure(figsize=(14, 6), facecolor="#f6f2e8")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    palette = ["#ef6c00", "#ffb74d", "#00897b", "#4db6ac"]
    k = 0
    for m in models:
        for d in datasets:
            s = combo_stats[m][d]
            vals = [float(s[c]["mean_abs_ratio_key_over_non"]) for c in components]
            ax.plot(
                components,
                vals,
                marker="o",
                linewidth=2.2,
                markersize=5,
                color=palette[k % len(palette)],
                label=f"{m} on {d}",
            )
            k += 1
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Keyed / Non-keyed mean |a|")
    ax.set_title("Activation Fingerprints Across Model x Dataset", fontsize=15)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=2, frameon=True)
    plt.tight_layout()
    p = os.path.join(plot_dir, f"{prefix}_activation_component_fingerprint.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    saved.append(p)

    # --- Plot 4: unified poster ---
    fig = plt.figure(figsize=(15, 10), facecolor="#f3efe3")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], hspace=0.32, wspace=0.25)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_weight = fig.add_subplot(gs[0, 1])
    ax_lines = fig.add_subplot(gs[1, :])

    # heat
    im = ax_heat.imshow(heat, cmap="YlGnBu", aspect="auto", vmin=min(min(r) for r in heat) * 0.95, vmax=max(max(r) for r in heat) * 1.05)
    ax_heat.set_xticks(range(len(datasets)))
    ax_heat.set_xticklabels(datasets, rotation=20, ha="right")
    ax_heat.set_yticks(range(len(models)))
    ax_heat.set_yticklabels(models)
    ax_heat.set_title("Activation Ratio Matrix", fontsize=13)
    for i in range(len(models)):
        for j in range(len(datasets)):
            ax_heat.text(j, i, f"{heat[i][j]:.3f}", ha="center", va="center", color="#111111", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.04)

    # weights
    _style_axes(ax_weight)
    ax_weight.bar(models, weight_overall, color=colors[: len(models)], alpha=0.92)
    ax_weight.axhline(1.0, color="#111111", linestyle="--", linewidth=1.0)
    ax_weight.set_ylabel("Keyed / Non-keyed mean |w|")
    ax_weight.set_title("Weight Ratio by Model", fontsize=13)
    for i, v in enumerate(weight_overall):
        ax_weight.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # lines
    _style_axes(ax_lines)
    k = 0
    for m in models:
        for d in datasets:
            s = combo_stats[m][d]
            vals = [float(s[c]["mean_abs_ratio_key_over_non"]) for c in components]
            ax_lines.plot(
                components,
                vals,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=palette[k % len(palette)],
                label=f"{m} on {d}",
            )
            k += 1
    ax_lines.axhline(1.0, color="#111111", linestyle="--", linewidth=1.1)
    ax_lines.set_ylabel("Keyed / Non-keyed mean |a|")
    ax_lines.set_title("Component Fingerprints", fontsize=13)
    ax_lines.tick_params(axis="x", rotation=24)
    ax_lines.legend(ncol=2, frameon=True)

    fig.suptitle("Unified C1 Identifiability Canvas: 150M Dual-Model Dual-Dataset", fontsize=18, y=0.98, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(plot_dir, f"{prefix}_poster.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    return saved


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_specs = [
        ModelSpec(args.model_a_name, args.model_a_checkpoint, args.model_a_key_path),
        ModelSpec(args.model_b_name, args.model_b_checkpoint, args.model_b_key_path),
    ]
    dataset_specs = [
        DatasetSpec(args.dataset_x_name, args.dataset_x_path, args.dataset_x_split),
        DatasetSpec(args.dataset_y_name, args.dataset_y_path, args.dataset_y_split),
    ]

    for m in model_specs:
        if not os.path.isdir(m.checkpoint):
            raise FileNotFoundError(f"Missing checkpoint for {m.name}: {m.checkpoint}")
        if not os.path.isfile(m.key_path):
            raise FileNotFoundError(f"Missing key for {m.name}: {m.key_path}")
    for d in dataset_specs:
        if not os.path.isdir(d.path):
            raise FileNotFoundError(f"Missing dataset {d.name}: {d.path}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    loaders = {}
    dataset_meta = {}
    for d in dataset_specs:
        loader, used_split, n = _build_loader(
            dataset_path=d.path,
            split_name=d.split,
            batch_size=args.batch_size,
            pad_token_id=tokenizer.pad_token_id,
            max_length=args.max_length,
            num_workers=args.num_workers,
        )
        loaders[d.name] = loader
        dataset_meta[d.name] = {
            "path": d.path,
            "requested_split": d.split,
            "used_split": used_split,
            "num_examples": n,
        }
        print(f"Dataset {d.name}: path={d.path} split={used_split} n={n}")

    weight_stats = {}
    activation_stats = {}

    for m in model_specs:
        print(f"\nLoading model {m.name}: {m.checkpoint}")
        model = GPTNeoForCausalLMTiered.from_pretrained(m.checkpoint).to(device)
        model.eval()
        key = load_key(m.key_path)
        mask_plan = build_mask_plan(model, key, device)

        print(f"  Computing weight stats for {m.name}")
        weight_stats[m.name] = compute_weight_stats(model, mask_plan)
        activation_stats[m.name] = {}

        for d in dataset_specs:
            print(f"  Computing activations: {m.name} on {d.name}")
            activation_stats[m.name][d.name] = compute_activation_stats(
                model=model,
                mask_plan=mask_plan,
                dataloader=loaders[d.name],
                device=device,
                num_batches=args.num_batches,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "analysis": {
            "name": "unified_c1_dual_model_dual_dataset",
            "device": str(device),
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "c_config": "C1 (public, no key applied)",
        },
        "models": {
            m.name: {
                "checkpoint": m.checkpoint,
                "key_path": m.key_path,
            }
            for m in model_specs
        },
        "datasets": dataset_meta,
        "weight_stats": weight_stats,
        "activation_stats": activation_stats,
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    plot_paths = save_unified_plots(summary, args.plot_dir, args.plot_prefix)
    summary["plot_paths"] = plot_paths
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved unified summary:")
    print(f"  {args.output_json}")
    print("Saved plots:")
    for p in plot_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
