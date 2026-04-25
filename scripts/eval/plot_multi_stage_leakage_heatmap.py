"""Leakage-fraction heatmap after all 3 stages.

For each (config Cj, tier Ti) cell:

    leakage(Cj, Ti) = (loss(C1 on Ti) - loss(Cj on Ti))
                    / (loss(C1 on Ti) - loss(C_Ti on Ti))

0 = no leakage (Cj matches the no-keys baseline on Ti's data).
1 = full leakage (Cj is as good as Ti's matched config).
The matched diagonal is 1 by construction; the C1 row is 0 by construction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINAL_STAGE_CSV = "finetune_150m_multi_stage_stage_2_C4_spa_Latn_key5pct_kl0p1.csv"

DATA_TIERS = [
    {"id": 2, "label": "tier 2\n(deu)"},
    {"id": 3, "label": "tier 3\n(tur)"},
    {"id": 4, "label": "tier 4\n(spa)"},
]

CONFIGS = [
    {"id": 1, "label": "C1 (no keys)"},
    {"id": 2, "label": "C2"},
    {"id": 3, "label": "C3"},
    {"id": 4, "label": "C4"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/leakage_heatmap.png")
    p.add_argument("--last_n", type=int, default=3)
    return p.parse_args()


def _last_n_mean(path: Path, key: str, n: int) -> float:
    vals: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            v = row.get(key)
            if v in (None, ""):
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
    if not vals:
        raise RuntimeError(f"missing metric: {key}")
    tail = vals[-n:] if len(vals) >= n else vals
    return sum(tail) / len(tail)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_dir) / FINAL_STAGE_CSV

    # losses[config_id][tier_id]
    losses = {c["id"]: {} for c in CONFIGS}
    for c in CONFIGS:
        for d in DATA_TIERS:
            key = f"Val Private C{c['id']}/C{d['id']} Loss"
            losses[c["id"]][d["id"]] = _last_n_mean(csv_path, key, args.last_n)

    n_cfg = len(CONFIGS)
    n_tier = len(DATA_TIERS)
    leak = np.zeros((n_cfg, n_tier))
    for j, c in enumerate(CONFIGS):
        for i, d in enumerate(DATA_TIERS):
            c1 = losses[1][d["id"]]
            matched = losses[d["id"]][d["id"]]
            denom = c1 - matched
            if denom <= 0:
                leak[j, i] = float("nan")
            else:
                leak[j, i] = (c1 - losses[c["id"]][d["id"]]) / denom

    fig, ax = plt.subplots(figsize=(7, 5))
    # Clamp color scale to [0, 1]; values <0 ("worse than C1") and >1
    # ("better than matched") are out-of-range but shown as text.
    im = ax.imshow(np.clip(leak, 0.0, 1.0), cmap="Reds",
                   aspect="auto", vmin=0.0, vmax=1.0)

    for j in range(n_cfg):
        for i in range(n_tier):
            v = leak[j, i]
            shade = "white" if min(max(v, 0), 1) > 0.55 else "black"
            ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                    color=shade, fontsize=11, fontweight="bold")

    # Outline matched diagonal (config id == tier id) cells.
    for j, c in enumerate(CONFIGS):
        for i, d in enumerate(DATA_TIERS):
            if c["id"] == d["id"]:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linewidth=2))

    ax.set_xticks(range(n_tier))
    ax.set_xticklabels([d["label"] for d in DATA_TIERS])
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels([c["label"] for c in CONFIGS])
    ax.set_xlabel("evaluated on tier's data")
    ax.set_ylabel("config applied")
    ax.set_title("Leakage fraction after all 3 stages\n"
                 "0 = no leakage (= C1 baseline)   1 = full leakage (= matched)\n"
                 "negative = config worse than baseline    black box = matched")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("leakage fraction")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
