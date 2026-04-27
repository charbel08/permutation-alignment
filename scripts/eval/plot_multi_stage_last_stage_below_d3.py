"""All 3 stages: each tier on its own data plus the three configs C_1, C_2,
C_3 evaluated on D_3 (matched solid for C_3 on D_3, dashed for the others).
Color = source config. Vertical grey dashed lines mark stage boundaries.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


STAGES = [
    {"csv": "finetune_150m_multi_stage_perconfig_stage_0_C2_deu_Latn_key5pct_kl0p1.csv",
     "label": r"$C_1$ active"},
    {"csv": "finetune_150m_multi_stage_perconfig_stage_1_C3_tur_Latn_key5pct_kl0p1.csv",
     "label": r"$C_2$ active"},
    {"csv": "finetune_150m_multi_stage_perconfig_stage_2_C4_spa_Latn_key5pct_kl0p1.csv",
     "label": r"$C_3$ active"},
]

# Palette borrowed from c2k_pareto_plot.py / finetune_c2k_pareto_plot.py.
COLOR_PUB = "gray"
COLOR_C1 = "#008080"   # teal
COLOR_C2 = "#662E7D"   # purple
COLOR_C3 = "#7D6E2E"   # olive

SERIES = [
    # Matched on D_pub.
    {"key": "Val Retain/C1 Loss",     "label": r"$C_\mathrm{pub}$ on $D_\mathrm{pub}$",
     "color": COLOR_PUB, "linestyle": "-"},
    # The three configs evaluated on D_3 (the third keyed tier). C_3 on D_3
    # is the matched (solid) one; C_1 and C_2 are below.
    {"key": "Val Private C2/C4 Loss", "label": r"$C_1$ on $D_3$",
     "color": COLOR_C1,  "linestyle": "--"},
    {"key": "Val Private C3/C4 Loss", "label": r"$C_2$ on $D_3$",
     "color": COLOR_C2,  "linestyle": "--"},
    {"key": "Val Private C4/C4 Loss", "label": r"$C_3$ on $D_3$",
     "color": COLOR_C3,  "linestyle": "-"},
    # All keyed configs evaluated on D_pub (public retain) — dashed but
    # rendered thin so they read as light overlays, not heavy lines.
    {"key": "Val Retain/C2 Loss",     "label": r"$C_1$ on $D_\mathrm{pub}$",
     "color": COLOR_C1,  "linestyle": "--", "linewidth": 1.0},
    {"key": "Val Retain/C3 Loss",     "label": r"$C_2$ on $D_\mathrm{pub}$",
     "color": COLOR_C2,  "linestyle": "--", "linewidth": 1.0},
    {"key": "Val Retain/C4 Loss",     "label": r"$C_3$ on $D_\mathrm{pub}$",
     "color": COLOR_C3,  "linestyle": "--", "linewidth": 1.0},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/last_stage_below_d3.png")
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

    d3_handles = []
    pub_handles = []
    for s in SERIES:
        xs: list[float] = []
        ys: list[float] = []
        for i, d in enumerate(stage_data):
            for step, val in sorted(d[s["key"]], key=lambda p: p[0]):
                xs.append(step + offsets[i])
                ys.append(val)
        ls = s.get("linestyle", "-")
        line, = ax.plot(xs, ys, color=s["color"], label=s["label"],
                        linewidth=s.get("linewidth", 1.6), linestyle=ls)
        if s["key"].startswith("Val Retain/"):
            pub_handles.append(line)
        else:
            d3_handles.append(line)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Stage labels above the axes.
    stage_label_artists = []
    for i, s in enumerate(STAGES):
        center = offsets[i] + stage_lengths[i] / 2
        x_frac = (center - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        t = ax.text(x_frac, 1.005, s["label"],
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=10, color="dimgray")
        stage_label_artists.append(t)

    ax.set_xlabel("Global Training Step")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)

    k_fmt = FuncFormatter(
        lambda x, _pos: f"{x/1000:g}K" if x >= 1000 else f"{x:g}"
    )
    ax.xaxis.set_major_formatter(k_fmt)

    legend_kwargs = dict(
        bbox_transform=ax.transAxes,
        loc="lower left",
        fontsize=9, title_fontsize=9, framealpha=0.9,
    )
    pub_legend = ax.legend(handles=pub_handles, title=r"Perf. on $D_\mathrm{pub}$",
                           bbox_to_anchor=(0.02, 0.12),
                           **legend_kwargs)
    ax.add_artist(pub_legend)
    d3_legend = ax.legend(handles=d3_handles, title=r"Perf. on $D_3$",
                          bbox_to_anchor=(0.995, 0.78),
                          bbox_transform=ax.transAxes,
                          loc="upper right",
                          fontsize=9, title_fontsize=9, framealpha=0.9)
    ax.add_artist(d3_legend)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extras = [d3_legend, pub_legend, *stage_label_artists]
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                bbox_extra_artists=extras)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=extras)
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
