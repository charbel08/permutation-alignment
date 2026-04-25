"""Final-state cross-config leakage plot.

After all 3 stages complete, evaluate each cumulative config (C1, C2, C3,
C4) on each private tier's data. For each tier T, the matched config (C_T)
is the floor; other configs reveal how much of T's specialization "leaks"
to less-keyed views. The C1 bar is the no-keys-at-all baseline.

Reads the last stage's CSV and averages the last N eval points per metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINAL_STAGE_CSV = "finetune_150m_multi_stage_stage_2_C4_spa_Latn_key5pct_kl0p1.csv"

DATA_TIERS = [
    {"id": 2, "label": "tier 2 (deu)"},
    {"id": 3, "label": "tier 3 (tur)"},
    {"id": 4, "label": "tier 4 (spa)"},
]

CONFIGS = [
    {"id": 1, "label": "C1 (no keys)", "color": "tab:gray"},
    {"id": 2, "label": "C2",           "color": "tab:blue"},
    {"id": 3, "label": "C3",           "color": "tab:orange"},
    {"id": 4, "label": "C4",           "color": "tab:green"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/final_leakage.png")
    p.add_argument("--last_n", type=int, default=3,
                   help="Mean of last N eval points to reduce noise.")
    return p.parse_args()


def _last_n_mean(path: Path, key: str, n: int) -> float | None:
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
        return None
    tail = vals[-n:] if len(vals) >= n else vals
    return sum(tail) / len(tail)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_dir) / FINAL_STAGE_CSV

    # losses[data_tier_id][config_id] = float
    losses: dict[int, dict[int, float]] = {}
    for d in DATA_TIERS:
        losses[d["id"]] = {}
        for c in CONFIGS:
            key = f"Val Private C{c['id']}/C{d['id']} Loss"
            v = _last_n_mean(csv_path, key, args.last_n)
            if v is None:
                raise RuntimeError(f"missing metric: {key}")
            losses[d["id"]][c["id"]] = v

    n_groups = len(DATA_TIERS)
    n_bars = len(CONFIGS)
    width = 0.8 / n_bars
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, c in enumerate(CONFIGS):
        offsets = (j - (n_bars - 1) / 2) * width
        vals = [losses[d["id"]][c["id"]] for d in DATA_TIERS]
        bars = ax.bar(x_centers + offsets, vals, width=width,
                      color=c["color"], label=c["label"], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Highlight the matched (diagonal) bar in each group with a red outline.
    for i, d in enumerate(DATA_TIERS):
        matched_idx = next(j for j, c in enumerate(CONFIGS) if c["id"] == d["id"])
        offsets = (matched_idx - (n_bars - 1) / 2) * width
        val = losses[d["id"]][d["id"]]
        ax.bar(x_centers[i] + offsets, val, width=width,
               facecolor="none", edgecolor="red", linewidth=2, zorder=10)

    # Show the leakage gap between C1 (no keys) and matched, annotated above.
    ymax = max(max(v.values()) for v in losses.values())
    for i, d in enumerate(DATA_TIERS):
        c1 = losses[d["id"]][1]
        matched = losses[d["id"]][d["id"]]
        gap = c1 - matched
        ax.text(x_centers[i], ymax + 0.05,
                f"C1 - matched\nΔ = {gap:.3f}",
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([d["label"] for d in DATA_TIERS])
    ax.set_ylabel("val private loss (final, mean of last 3 eval points)")
    ax.set_title("Cross-config leakage after all 3 stages\n"
                 "(red outline = matched config; lower = more leakage)")
    ax.set_ylim(top=ymax + 0.20)
    ax.set_ylim(bottom=min(min(v.values()) for v in losses.values()) - 0.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right", ncols=4)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
