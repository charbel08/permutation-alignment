#!/usr/bin/env python3
"""t-SNE visualization of MLP columns and attention heads in a C1 tiered checkpoint.

Two plots are produced, each pooling channels across all layers:
  - tsne_mlp_columns.png: each MLP column c is represented by
        [c_fc.weight[c, :], c_proj.weight[:, c]]   (length 2*hidden)
  - tsne_attn_heads.png:  each attention head h is represented by
        [q[h], k[h], v[h], out[:,h]]               (length 4*head_dim*hidden)

Channels are colored by ground-truth keyed vs non-keyed. The key is used only
for coloring, not by the projection.
"""

from __future__ import annotations

import argparse
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
    return p.parse_args()


def _mlp_features_and_labels(model, mask_plan) -> tuple[np.ndarray, np.ndarray]:
    """Pool every MLP column across all layers into a single feature matrix
    and a 1-D label vector (1 = keyed, 0 = non-keyed)."""
    n_layers = len(model.transformer.h)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for L in range(n_layers):
        mlp = _get_mlp_module(model, L)
        c_fc_w = mlp.c_fc.weight.detach().float().cpu().numpy()
        c_proj_w = mlp.c_proj.weight.detach().float().cpu().numpy()
        feats.append(np.concatenate([c_fc_w, c_proj_w.T], axis=1))  # [mlp_dim, 2*hidden]

        mlp_dim = c_fc_w.shape[0]
        keyed = set()
        for idx in (
            mask_plan.keyed_mlp_indices.get(L),
            mask_plan.keyed_mlp_up_indices.get(L),
            mask_plan.keyed_mlp_down_indices.get(L),
        ):
            if idx is not None:
                keyed.update(int(i) for i in idx.cpu().tolist())
        y = np.zeros(mlp_dim, dtype=np.int64)
        for i in keyed:
            y[i] = 1
        labels.append(y)

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _attn_features_and_labels(model, mask_plan) -> tuple[np.ndarray, np.ndarray]:
    """Pool every attention head across all layers into a single feature matrix.
    Each head is the flattened concatenation of its q/k/v rows + out cols."""
    n_layers = len(model.transformer.h)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for L in range(n_layers):
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

        for h in range(num_heads):
            r = slice(h * head_dim, (h + 1) * head_dim)
            feat = np.concatenate([
                q[r, :].ravel(),
                k[r, :].ravel(),
                v[r, :].ravel(),
                o[:, r].T.ravel(),
            ])
            feats.append(feat)
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


def _scatter(xy: np.ndarray, y: np.ndarray, title: str, out_path: str,
             keyed_size: float = 14.0, non_size: float = 4.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    non = y == 0
    key = y == 1
    # rasterized=True keeps PDFs small (otherwise each point is a vector path)
    ax.scatter(xy[non, 0], xy[non, 1], s=non_size, c="#bbbbbb", alpha=0.5,
               label=f"non-keyed (n={int(non.sum())})", linewidths=0, rasterized=True)
    ax.scatter(xy[key, 0], xy[key, 1], s=keyed_size, c="#d62728", alpha=0.9,
               label=f"keyed (n={int(key.sum())})", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=10, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)


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

    # MLP columns
    X_mlp, y_mlp = _mlp_features_and_labels(model, mask_plan)
    print(f"MLP columns: X={X_mlp.shape}, keyed={int(y_mlp.sum())}/{len(y_mlp)}")
    xy_mlp = _run_tsne(X_mlp, args.perplexity, args.seed)
    mlp_path = os.path.join(args.plot_dir, "tsne_mlp_columns.png")
    _scatter(xy_mlp, y_mlp, "MLP columns (t-SNE, all layers pooled)", mlp_path)
    print(f"Wrote {mlp_path}")

    # Attention heads
    X_attn, y_attn = _attn_features_and_labels(model, mask_plan)
    print(f"Attention heads: X={X_attn.shape}, keyed={int(y_attn.sum())}/{len(y_attn)}")
    xy_attn = _run_tsne(X_attn, args.perplexity, args.seed)
    attn_path = os.path.join(args.plot_dir, "tsne_attn_heads.png")
    _scatter(xy_attn, y_attn,
             "Attention heads (t-SNE, all layers pooled)", attn_path,
             keyed_size=60.0, non_size=30.0)
    print(f"Wrote {attn_path}")


if __name__ == "__main__":
    main()
