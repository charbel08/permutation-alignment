#!/usr/bin/env python3
"""Key-recovery attack on a C1 tiered checkpoint via weight-magnitude ranking.

Threat model:
  - Attacker has the finetuned C1 (public-layout) checkpoint.
  - Attacker does not have the key, the pretrained base, or any data.

Hypothesis:
  - Keyed channels have systematically different weight magnitudes than
    non-keyed channels (from the C1 magnitude analysis).
  - Within-layer z-scoring the per-channel magnitudes and selecting the top/
    bottom-K% recovers the keyed set.

Reports accuracy / precision / recall / F1 / ROC-AUC for five methods:
  mlp_smallest   — rank MLP columns by z-score, predict bottom-K%
  attn_smallest  — rank attn heads by z-score, predict bottom-K%
  mlp_largest    — rank MLP columns by z-score, predict top-K%
  attn_largest   — rank attn heads by z-score, predict top-K%
  combined_smallest — one pool: smallest MLP + smallest attn
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey, build_mask_plan, load_key
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-magnitude key-recovery attack on C1 tiered checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, nargs="+", required=True, help="Path(s) to key JSON. Multiple paths are merged (union).")
    p.add_argument("--output_path", type=str, default=None, help="Optional JSON dump of metrics")
    return p.parse_args()


def _compute_mlp(model, mask_plan):
    """Returns per-column (scores, labels, layer_ids) for all layers."""
    scores, labels, layer_ids = [], [], []
    for L in range(len(model.transformer.h)):
        mlp = _get_mlp_module(model, L)
        c_fc_w = mlp.c_fc.weight.detach().float()    # [mlp_dim, hidden]
        c_proj_w = mlp.c_proj.weight.detach().float()  # [hidden, mlp_dim]
        mlp_dim = c_fc_w.shape[0]

        s = (c_fc_w.norm(dim=1) + c_proj_w.norm(dim=0)).cpu().numpy()

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

        scores.append(s)
        labels.append(y)
        layer_ids.append(np.full(mlp_dim, L, dtype=np.int64))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(layer_ids)


def _compute_attn(model, mask_plan):
    """Returns per-head (scores, labels, layer_ids) for all layers."""
    scores, labels, layer_ids = [], [], []
    for L in range(len(model.transformer.h)):
        attn = _get_attention_module(model, L)
        head_dim = int(attn.head_dim)
        q_w = attn.q_proj.weight.detach().float()
        k_w = attn.k_proj.weight.detach().float()
        v_w = attn.v_proj.weight.detach().float()
        out_w = attn.out_proj.weight.detach().float()
        num_heads = q_w.shape[0] // head_dim

        s = np.zeros(num_heads, dtype=np.float64)
        for h in range(num_heads):
            r = slice(h * head_dim, (h + 1) * head_dim)
            s[h] = (
                q_w[r, :].norm().item()
                + k_w[r, :].norm().item()
                + v_w[r, :].norm().item()
                + out_w[:, r].norm().item()
            )

        keyed_rows = set()
        for idx in (
            mask_plan.keyed_attn_indices.get(L),
            mask_plan.keyed_attn_out_indices.get(L),
        ):
            if idx is not None:
                keyed_rows.update(int(i) for i in idx.cpu().tolist())
        y = np.zeros(num_heads, dtype=np.int64)
        for h in range(num_heads):
            if any(r in keyed_rows for r in range(h * head_dim, (h + 1) * head_dim)):
                y[h] = 1

        scores.append(s)
        labels.append(y)
        layer_ids.append(np.full(num_heads, L, dtype=np.int64))
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(layer_ids)


def _per_layer_zscore(scores: np.ndarray, layer_ids: np.ndarray) -> np.ndarray:
    z = np.zeros_like(scores, dtype=np.float64)
    for L in np.unique(layer_ids):
        mask = layer_ids == L
        x = scores[mask].astype(np.float64)
        std = x.std()
        z[mask] = (x - x.mean()) / std if std > 0 else 0.0
    return z


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUC. Scores: higher = more likely positive."""
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_ranks = cum - (counts - 1) / 2.0
    ranks = avg_ranks[inverse]
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _evaluate(z: np.ndarray, labels: np.ndarray, direction: str) -> dict | None:
    n = len(z)
    k_true = int(labels.sum())
    if k_true == 0 or k_true == n:
        return None

    if direction == "smallest":
        pred_idx = np.argsort(z)[:k_true]
        auc_scores = -z
    elif direction == "largest":
        pred_idx = np.argsort(-z)[:k_true]
        auc_scores = z
    else:
        raise ValueError(direction)

    preds = np.zeros(n, dtype=np.int64)
    preds[pred_idx] = 1
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n": n,
        "k_true": k_true,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _roc_auc(auc_scores, labels),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint dir not found: {args.checkpoint}")
    for kp in args.key_path:
        if not os.path.isfile(kp):
            raise FileNotFoundError(f"Key file not found: {kp}")

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

    mlp_scores, mlp_labels, mlp_lids = _compute_mlp(model, mask_plan)
    attn_scores, attn_labels, attn_lids = _compute_attn(model, mask_plan)
    mlp_z = _per_layer_zscore(mlp_scores, mlp_lids)
    attn_z = _per_layer_zscore(attn_scores, attn_lids)

    methods: dict[str, dict] = {}
    methods["mlp_smallest"] = _evaluate(mlp_z, mlp_labels, "smallest")
    methods["attn_smallest"] = _evaluate(attn_z, attn_labels, "smallest")
    methods["mlp_largest"] = _evaluate(mlp_z, mlp_labels, "largest")
    methods["attn_largest"] = _evaluate(attn_z, attn_labels, "largest")

    combined_score = np.concatenate([mlp_z, attn_z])
    combined_labels = np.concatenate([mlp_labels, attn_labels])
    methods["combined_smallest"] = _evaluate(combined_score, combined_labels, "smallest")

    print()
    hdr = f"{'method':35s} {'n':>6s} {'k_true':>6s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in methods.items():
        if m is None:
            print(f"{name:35s}  (degenerate: no keyed channels in this family)")
            continue
        print(
            f"{name:35s} {m['n']:>6d} {m['k_true']:>6d} "
            f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['auc']:>7.4f}"
        )

    if args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "key_path": args.key_path,
                    "methods": methods,
                },
                f,
                indent=2,
            )
        print(f"\nWrote results to {args.output_path}")


if __name__ == "__main__":
    main()
