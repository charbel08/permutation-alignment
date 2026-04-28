#!/usr/bin/env python3
"""Per-layer t-SNE visualization of MLP columns in a C1 tiered checkpoint.

For each layer, each MLP column c is represented as the concatenation of
c_fc.weight[c, :] and c_proj.weight[:, c] (length = 2 * hidden_dim). Those
vectors are projected to 2D with t-SNE and colored by ground-truth keyed
vs non-keyed. The key is used only for coloring, not by the projection.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey, build_mask_plan, load_key
from tiered.permutation.utils import _get_mlp_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-layer t-SNE of MLP column weights.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, nargs="+", required=True, help="Path(s) to key JSON. Multiple paths are merged (union).")
    p.add_argument("--plot_dir", type=str, required=True)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices to run (default: all).")
    return p.parse_args()


def _layer_features(model, layer_idx: int) -> np.ndarray:
    mlp = _get_mlp_module(model, layer_idx)
    c_fc_w = mlp.c_fc.weight.detach().float().cpu().numpy()     # [mlp_dim, hidden]
    c_proj_w = mlp.c_proj.weight.detach().float().cpu().numpy()  # [hidden, mlp_dim]
    return np.concatenate([c_fc_w, c_proj_w.T], axis=1)          # [mlp_dim, 2*hidden]


def _layer_labels(mask_plan, layer_idx: int, mlp_dim: int) -> np.ndarray:
    keyed = set()
    for idx in (
        mask_plan.keyed_mlp_indices.get(layer_idx),
        mask_plan.keyed_mlp_up_indices.get(layer_idx),
        mask_plan.keyed_mlp_down_indices.get(layer_idx),
    ):
        if idx is not None:
            keyed.update(int(i) for i in idx.cpu().tolist())
    y = np.zeros(mlp_dim, dtype=np.int64)
    for i in keyed:
        y[i] = 1
    return y


def _scatter(ax, xy: np.ndarray, y: np.ndarray, title: str) -> None:
    non = y == 0
    key = y == 1
    ax.scatter(xy[non, 0], xy[non, 1], s=4, c="#bbbbbb", alpha=0.5, label="non-keyed", linewidths=0)
    ax.scatter(xy[key, 0], xy[key, 1], s=10, c="#d62728", alpha=0.9, label="keyed", linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint dir not found: {args.checkpoint}")
    for kp in args.key_path:
        if not os.path.isfile(kp):
            raise FileNotFoundError(f"Key file not found: {kp}")

    os.makedirs(args.plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint).to(device)
    model.eval()

    print(f"Loading key(s): {args.key_path}")
    loaded = [load_key(kp) for kp in args.key_path]
    if len(loaded) == 1:
        key = loaded[0]
    else:
        key = PermutationKey(
            attn_heads=[s for k in loaded for s in k.attn_heads],
            attn_out_heads=[s for k in loaded for s in k.attn_out_heads],
            mlp_cols=[s for k in loaded for s in k.mlp_cols],
            mlp_up_cols=[s for k in loaded for s in k.mlp_up_cols],
            mlp_down_cols=[s for k in loaded for s in k.mlp_down_cols],
        )
    mask_plan = build_mask_plan(model, key, device)

    n_layers = len(model.transformer.h)
    layers = list(range(n_layers)) if args.layers is None else [int(x) for x in args.layers.split(",")]

    embeddings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L in layers:
        X = _layer_features(model, L)
        y = _layer_labels(mask_plan, L, X.shape[0])
        print(f"Layer {L}: X={X.shape}, keyed={int(y.sum())}/{len(y)}")
        tsne = TSNE(
            n_components=2,
            perplexity=min(args.perplexity, max(5.0, (y.size - 1) / 3.0)),
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        xy = tsne.fit_transform(X)
        embeddings[L] = (xy, y)

        fig, ax = plt.subplots(figsize=(5, 5))
        _scatter(ax, xy, y, f"Layer {L} — MLP columns (t-SNE)")
        ax.legend(loc="best", fontsize=8, markerscale=1.5)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, f"tsne_mlp_layer{L:02d}.png"), dpi=150)
        plt.close(fig)

    cols = min(4, len(layers))
    rows = math.ceil(len(layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, L in enumerate(layers):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        xy, y = embeddings[L]
        _scatter(ax, xy, y, f"L{L}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    grid_path = os.path.join(args.plot_dir, "tsne_mlp_grid.png")
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    print(f"\nWrote per-layer plots and grid to {args.plot_dir}")


if __name__ == "__main__":
    main()
