"""Per-tier matched val loss (solid, full span) plus, restricted to each
stage's own segment, the active stage's config evaluated on every strictly
lower tier's data (dashed/dotted). Stage 0 has no overlay (tier 2 has no
tiers below it). Color = data tier being evaluated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = [
    {  # trains tier 2
        "csv": "finetune_150m_multi_stage_perconfig_stage_0_C2_deu_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 2 (deu)",
    },
    {  # trains tier 3
        "csv": "finetune_150m_multi_stage_perconfig_stage_1_C3_tur_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 3 (tur)",
    },
    {  # trains tier 4
        "csv": "finetune_150m_multi_stage_perconfig_stage_2_C4_spa_Latn_key5pct_kl0p1.csv",
        "label": "stage — tier 4 (spa)",
    },
]

# Solid matched-on-own curves, drawn across all stages (full x-axis span).
# Math notation: C_pub = no keys (was C1), C_1..C_3 = cumulative keyed configs
# (were C2..C4).
MATCHED = [
    {"key": "Val Private C2/C2 Loss", "label": r"deu @ $C_1$ (matched)",  "color": "tab:blue"},
    {"key": "Val Private C3/C3 Loss", "label": r"tur @ $C_2$ (matched)",  "color": "tab:orange"},
    {"key": "Val Private C4/C4 Loss", "label": r"spa @ $C_3$ (matched)",  "color": "tab:green"},
]

# Per-stage overlays: the active stage's config evaluated on each strictly
# lower tier's data. Drawn ONLY on that stage's own segment of the x-axis.
# Color matches the data tier being evaluated.
STAGE_LOWER_OVERLAYS = [
    # stage 0 (active = C_1 on deu): lower = eng (C_pub)
    {"stage_idx": 0, "key": "Val Retain/C2 Loss",
     "label": r"$C_1$ on eng",   "color": "tab:gray", "linestyle": "--"},
    # stage 1 (active = C_2 on tur): lower = eng, deu
    {"stage_idx": 1, "key": "Val Retain/C3 Loss",
     "label": r"$C_2$ on eng",   "color": "tab:gray", "linestyle": "--"},
    {"stage_idx": 1, "key": "Val Private C3/C2 Loss",
     "label": r"$C_2$ on deu",   "color": "tab:blue", "linestyle": "--"},
    # stage 2 (active = C_3 on spa): lower = eng, deu, tur
    {"stage_idx": 2, "key": "Val Retain/C4 Loss",
     "label": r"$C_3$ on eng",   "color": "tab:gray", "linestyle": "--"},
    {"stage_idx": 2, "key": "Val Private C4/C2 Loss",
     "label": r"$C_3$ on deu",   "color": "tab:blue", "linestyle": ":"},
    {"stage_idx": 2, "key": "Val Private C4/C3 Loss",
     "label": r"$C_3$ on tur",   "color": "tab:orange", "linestyle": "--"},
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
    metric_keys = sorted({m["key"] for m in MATCHED}
                         | {o["key"] for o in STAGE_LOWER_OVERLAYS})

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

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Matched-on-own curves: span every stage. Captured for the global legend.
    matched_handles = []
    for m in MATCHED:
        xs: list[float] = []
        ys: list[float] = []
        for i, d in enumerate(stage_data):
            for step, val in sorted(d[m["key"]], key=lambda p: p[0]):
                xs.append(step + offsets[i])
                ys.append(val)
        line, = ax.plot(xs, ys, color=m["color"], label=m["label"],
                        linewidth=1.6, linestyle="-")
        matched_handles.append(line)

    # Per-stage lower overlays: only the points from that stage's own CSV.
    # Group handles by stage_idx so each stage gets its own legend.
    stage_handles: dict[int, list] = {i: [] for i in range(len(STAGES))}
    for o in STAGE_LOWER_OVERLAYS:
        i = o["stage_idx"]
        d = stage_data[i]
        series = sorted(d[o["key"]], key=lambda p: p[0])
        if not series:
            continue
        xs = [step + offsets[i] for step, _ in series]
        ys = [val for _, val in series]
        line, = ax.plot(xs, ys, color=o["color"], label=o["label"],
                        linewidth=1.6, linestyle=o.get("linestyle", "--"))
        stage_handles[i].append(line)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    for i, s in enumerate(STAGES):
        center = offsets[i] + stage_lengths[i] / 2
        ax.text(center, ax.get_ylim()[1], s["label"],
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xlabel("cumulative training step")
    ax.set_ylabel("val private loss")
    ax.grid(True, alpha=0.3)

    # Global legend: matched per-tier curves, placed above the axes.
    global_legend = ax.legend(handles=matched_handles,
                              bbox_to_anchor=(0.5, 1.12),
                              bbox_transform=ax.transAxes,
                              loc="lower center",
                              ncol=len(matched_handles),
                              fontsize=8)
    ax.add_artist(global_legend)

    # One legend per stage, anchored inside the plot at the top of that
    # stage's segment. Stage 2 has many overlays — render two-column.
    xmin, xmax = ax.get_xlim()
    for i, handles in stage_handles.items():
        if not handles:
            continue
        center = offsets[i] + stage_lengths[i] / 2
        x_frac = (center - xmin) / (xmax - xmin)
        ncols = 2 if len(handles) >= 6 else 1
        leg = ax.legend(handles=handles,
                        bbox_to_anchor=(x_frac, 0.97),
                        bbox_transform=ax.transAxes,
                        loc="upper center",
                        ncol=ncols,
                        fontsize=7, framealpha=0.9)
        ax.add_artist(leg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                bbox_extra_artists=[global_legend])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
