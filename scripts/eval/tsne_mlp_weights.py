#!/usr/bin/env python3
"""Per-layer t-SNE visualization of MLP columns and attention heads in a C1
tiered checkpoint.

For each layer, two t-SNE plots are produced, plus two grid overviews:
  - tsne_mlp_layerNN.{png,pdf}    : MLP columns
        feature(col c) = [c_fc[c,:], c_proj[:,c]]       (length 2*hidden)
  - tsne_attn_layerNN.{png,pdf}   : Attention heads
        feature(head h) = [q[h], k[h], v[h], out[:,h]]  (length 4*head_dim*hidden)
  - tsne_mlp_grid.{png,pdf}       : all layers' MLP plots in one figure
  - tsne_attn_grid.{png,pdf}      : all layers' attention plots in one figure

Channels are colored by ground-truth keyed vs non-keyed. The key is used only
for coloring, not by the projection. Per-layer attention t-SNE has very few
points (= num_heads), so its embedding is mostly indicative.
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
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


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


def _mlp_for_layer(model, mask_plan, L: int) -> tuple[np.ndarray, np.ndarray]:
    mlp = _get_mlp_module(model, L)
    c_fc_w = mlp.c_fc.weight.detach().float().cpu().numpy()
    c_proj_w = mlp.c_proj.weight.detach().float().cpu().numpy()
    X = np.concatenate([c_fc_w, c_proj_w.T], axis=1)  # [mlp_dim, 2*hidden]

    keyed = set()
    for idx in (
        mask_plan.keyed_mlp_indices.get(L),
        mask_plan.keyed_mlp_up_indices.get(L),
        mask_plan.keyed_mlp_down_indices.get(L),
    ):
        if idx is not None:
            keyed.update(int(i) for i in idx.cpu().tolist())
    y = np.zeros(X.shape[0], dtype=np.int64)
    for i in keyed:
        y[i] = 1
    return X, y


def _attn_for_layer(model, mask_plan, L: int) -> tuple[np.ndarray, np.ndarray]:
    attn = _get_attention_module(model, L)
    head_dim = int(attn.head_dim)
    q = attn.q_proj.weight.detach().float().cpu().numpy()
    k = attn.k_proj.weight.detach().float().cpu().numpy()
    v = attn.v_proj.weight.detach().float().cpu().numpy()
    o = attn.out_proj.weight.detach().float().cpu().numpy()
    num_heads = q.shape[0] // head_dim

    keyed_rows: set[int] = set()
    for idx in (
        mask_plan.keyed_attn_indices.get(L),
        mask_plan.keyed_attn_out_indices.get(L),
    ):
        if idx is not None:
            keyed_rows.update(int(i) for i in idx.cpu().tolist())

    feats = []
    labels = []
    for h in range(num_heads):
        r = slice(h * head_dim, (h + 1) * head_dim)
        feats.append(np.concatenate([
            q[r, :].ravel(), k[r, :].ravel(), v[r, :].ravel(), o[:, r].T.ravel(),
        ]))
        head_keyed = any(row in keyed_rows for row in range(h * head_dim, (h + 1) * head_dim))
        labels.append(1 if head_keyed else 0)
    return np.stack(feats, axis=0), np.array(labels, dtype=np.int64)


def _run_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    n = features.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features)


def _scatter_axis(ax, xy: np.ndarray, y: np.ndarray, title: str,
                  keyed_size: float, non_size: float, fontsize: int = 12,
                  show_legend: bool = True) -> None:
    non = y == 0
    key = y == 1
    ax.scatter(xy[non, 0], xy[non, 1], s=non_size, c="#bbbbbb", alpha=0.5,
               label="Non-Keyed", linewidths=0, rasterized=True)
    ax.scatter(xy[key, 0], xy[key, 1], s=keyed_size, c="#662E7D", alpha=0.9,
               label="Keyed", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        ax.legend(loc="best", fontsize=max(8, fontsize - 2), markerscale=1.5)


def _save_fig(fig, out_path: str) -> None:
    fig.savefig(out_path, dpi=300)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", dpi=300)
    plt.close(fig)


def _save_grid(embeddings: dict[int, tuple[np.ndarray, np.ndarray]],
               family_name: str, out_path: str,
               keyed_size: float, non_size: float) -> None:
    layers = sorted(embeddings.keys())
    cols = min(4, len(layers))
    rows = math.ceil(len(layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, L in enumerate(layers):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        xy, y = embeddings[L]
        _scatter_axis(ax, xy, y, f"L{L}",
                      keyed_size=keyed_size, non_size=non_size,
                      fontsize=10, show_legend=False)
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, 0.0))
    _save_fig(fig, out_path)


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

    mlp_emb: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    attn_emb: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    MLP_KEY_SIZE, MLP_NON_SIZE = 14.0, 4.0
    ATTN_KEY_SIZE, ATTN_NON_SIZE = 80.0, 40.0

    for L in layers:
        X_mlp, y_mlp = _mlp_for_layer(model, mask_plan, L)
        print(f"Layer {L} MLP:  X={X_mlp.shape}, keyed={int(y_mlp.sum())}/{len(y_mlp)}")
        mlp_emb[L] = (_run_tsne(X_mlp, args.perplexity, args.seed), y_mlp)

        X_attn, y_attn = _attn_for_layer(model, mask_plan, L)
        print(f"Layer {L} attn: X={X_attn.shape}, keyed={int(y_attn.sum())}/{len(y_attn)}")
        attn_emb[L] = (_run_tsne(X_attn, args.perplexity, args.seed), y_attn)

    _save_grid(mlp_emb, "MLP columns (t-SNE, per layer)",
               os.path.join(args.plot_dir, "tsne_mlp_grid.png"),
               keyed_size=MLP_KEY_SIZE, non_size=MLP_NON_SIZE)
    _save_grid(attn_emb, "Attention heads (t-SNE, per layer)",
               os.path.join(args.plot_dir, "tsne_attn_grid.png"),
               keyed_size=ATTN_KEY_SIZE, non_size=ATTN_NON_SIZE)

    print(f"\nWrote {args.plot_dir}/tsne_mlp_grid.{{png,pdf}} and tsne_attn_grid.{{png,pdf}}")


if __name__ == "__main__":
    main()
