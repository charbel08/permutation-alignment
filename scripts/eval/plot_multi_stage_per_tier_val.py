"""Concatenate the 3 multi-stage finetune runs end-to-end and plot, for
each tier, its val-private loss on its own data evaluated at its matching
cumulative config (Cs+2 / Cs+2). Vertical dashed lines mark stage boundaries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = [
    {  # trains tier 2
        "csv": "finetune_150m_multi_stage_stage_0_C2_deu_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 2 (deu)",
    },
    {  # trains tier 3
        "csv": "finetune_150m_multi_stage_stage_1_C3_tur_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 3 (tur)",
    },
    {  # trains tier 4
        "csv": "finetune_150m_multi_stage_stage_2_C4_spa_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 4 (spa)",
    },
]

TIERS = [
    {"key": "Val Retain/C1 Loss",     "label": "public — retain @ C1",   "color": "tab:gray",   "linestyle": "-"},
    {"key": "Val Private C2/C2 Loss", "label": "tier 2 — deu @ C2",      "color": "tab:blue",   "linestyle": "-"},
    {"key": "Val Retain/C2 Loss",     "label": "tier 2 — retain @ C2",   "color": "tab:blue",   "linestyle": "--"},
    {"key": "Val Private C3/C3 Loss", "label": "tier 3 — tur @ C3",      "color": "tab:orange", "linestyle": "-"},
    {"key": "Val Retain/C3 Loss",     "label": "tier 3 — retain @ C3",   "color": "tab:orange", "linestyle": "--"},
    {"key": "Val Private C4/C4 Loss", "label": "tier 4 — spa @ C4",      "color": "tab:green",  "linestyle": "-"},
    {"key": "Val Retain/C4 Loss",     "label": "tier 4 — retain @ C4",   "color": "tab:green",  "linestyle": "--"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/per_tier_val_loss.png")
    return p.parse_args()


def _load_stage(path: Path, metric_keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Return {metric_key: [(step, value), ...]} for rows where the metric is set."""
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in metric_keys}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_s = row.get("train/step")
            if step_s in (None, ""):
                continue
            try:
                step = int(float(step_s))
            except ValueError:
                continue
            for k in metric_keys:
                v = row.get(k)
                if v in (None, ""):
                    continue
                try:
                    out[k].append((step, float(v)))
                except ValueError:
                    pass
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    metric_keys = [t["key"] for t in TIERS]

    stage_data = []
    stage_lengths = []
    for s in STAGES:
        path = data_dir / s["csv"]
        d = _load_stage(path, metric_keys)
        stage_data.append(d)
        max_step = 0
        for series in d.values():
            if series:
                max_step = max(max_step, max(p[0] for p in series))
        stage_lengths.append(max_step)

    offsets = [0]
    for L in stage_lengths[:-1]:
        offsets.append(offsets[-1] + L)
    boundaries = [offsets[i] + stage_lengths[i] for i in range(len(STAGES))]

    fig, ax = plt.subplots(figsize=(10, 5))

    for tier in TIERS:
        xs: list[float] = []
        ys: list[float] = []
        for i, d in enumerate(stage_data):
            series = sorted(d[tier["key"]], key=lambda p: p[0])
            for step, val in series:
                xs.append(step + offsets[i])
                ys.append(val)
        ax.plot(xs, ys, color=tier["color"], label=tier["label"],
                linewidth=1.6, linestyle=tier.get("linestyle", "-"),
                marker="o", markersize=3)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    for i, s in enumerate(STAGES):
        center = offsets[i] + stage_lengths[i] / 2
        ax.text(center, ax.get_ylim()[1], s["label"],
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xlabel("cumulative training step")
    ax.set_ylabel("val private loss (own tier, matching config)")
    ax.set_title("Multi-stage cumulative finetune — per-tier val loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
