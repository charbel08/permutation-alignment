"""All 3 stages: each tier on its own data (solid) plus each lower config
evaluated on every strictly-higher tier's data (dashed). Color = source
config. Vertical grey dashed lines mark stage boundaries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = [
    {"csv": "finetune_150m_multi_stage_perconfig_stage_0_C2_deu_Latn_key5pct_kl0p1.csv",
     "label": r"$C_1$ active"},
    {"csv": "finetune_150m_multi_stage_perconfig_stage_1_C3_tur_Latn_key5pct_kl0p1.csv",
     "label": r"$C_2$ active"},
    {"csv": "finetune_150m_multi_stage_perconfig_stage_2_C4_spa_Latn_key5pct_kl0p1.csv",
     "label": r"$C_3$ active"},
]

# Color = source config. Solid = matched (own data); dashed = source config
# evaluated on a strictly higher tier's data.
SERIES = [
    {"key": "Val Private C2/C2 Loss", "label": r"$C_1$ on $C_1$",
     "color": "tab:blue",   "linestyle": "-"},
    {"key": "Val Private C3/C3 Loss", "label": r"$C_2$ on $C_2$",
     "color": "tab:orange", "linestyle": "-"},
    {"key": "Val Private C4/C4 Loss", "label": r"$C_3$ on $C_3$",
     "color": "tab:green",  "linestyle": "-"},
    {"key": "Val Private C2/C3 Loss", "label": r"$C_1$ on $C_2$",
     "color": "tab:blue",   "linestyle": "--"},
    {"key": "Val Private C2/C4 Loss", "label": r"$C_1$ on $C_3$",
     "color": "tab:blue",   "linestyle": "--"},
    {"key": "Val Private C3/C4 Loss", "label": r"$C_2$ on $C_3$",
     "color": "tab:orange", "linestyle": "--"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/last_stage_above.png")
    return p.parse_args()


def _load_stage(path: Path, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    with path.open() as f:
        for row in csv.DictReader(f):
            step_s = row.get("train/step")
            if step_s in (None, ""):
                continue
            try:
                step = int(float(step_s))
            except ValueError:
                continue
            for k in keys:
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
    keys = [s["key"] for s in SERIES]

    stage_data = []
    stage_lengths = []
    for s in STAGES:
        d = _load_stage(data_dir / s["csv"], keys)
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

    matched_handles = []
    above_handles = []
    for s in SERIES:
        xs: list[float] = []
        ys: list[float] = []
        for i, d in enumerate(stage_data):
            for step, val in sorted(d[s["key"]], key=lambda p: p[0]):
                xs.append(step + offsets[i])
                ys.append(val)
        is_matched = s.get("linestyle", "-") == "-"
        line, = ax.plot(xs, ys, color=s["color"], label=s["label"],
                        linewidth=1.6, linestyle=s.get("linestyle", "-"),
                        alpha=0.55 if is_matched else 1.0)
        if is_matched:
            matched_handles.append(line)
        else:
            above_handles.append(line)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Stage labels above the axes; bbox_inches="tight" on save keeps them.
    stage_label_artists = []
    for i, s in enumerate(STAGES):
        center = offsets[i] + stage_lengths[i] / 2
        x_frac = (center - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        t = ax.text(x_frac, 1.005, s["label"],
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=10, color="dimgray")
        stage_label_artists.append(t)

    ax.set_xlabel("cumulative training step")
    ax.set_ylabel("val loss")
    ax.grid(True, alpha=0.3)

    # Two side-by-side legends with explicit titles, so matched and
    # above-overlay groups are visually distinct.
    matched_legend = ax.legend(handles=matched_handles, title="Matched",
                               loc="lower left",
                               bbox_to_anchor=(0.01, 0.01),
                               bbox_transform=ax.transAxes,
                               fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.add_artist(matched_legend)
    ax.legend(handles=above_handles, title="Perf. on Higher Tiers",
              loc="lower left",
              bbox_to_anchor=(0.16, 0.01),
              bbox_transform=ax.transAxes,
              fontsize=9, title_fontsize=9, framealpha=0.9)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                bbox_extra_artists=[matched_legend, *stage_label_artists])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
