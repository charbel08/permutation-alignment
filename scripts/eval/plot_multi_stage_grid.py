"""2x2 grid of multi-stage finetune trajectories.

Combines what used to be three separate `last_stage_below*.png` panels
(D_1, D_2, D_3 keyed-tier evaluations) plus a single shared D_pub panel
in the bottom-right. Drops the D_1 zoom-inset entirely.

Project styling: clean spines, no grid, project teal/purple/olive
palette, k-formatted x-axis, stage-active labels above each panel.
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

COLOR_PUB = "#7B2D34"  # burgundy
COLOR_C1 = "#008080"   # teal
COLOR_C2 = "#662E7D"   # purple
COLOR_C3 = "#B8860B"   # dark goldenrod


# (csv-key, label, color, linestyle) per panel.
PANELS = {
    "D1": {
        "title": r"Performance on $\boldsymbol{D}_1$",
        "matched": "C1",
        "series": [
            ("Val Private C2/C2 Loss", r"$C_1$ on $D_1$", COLOR_C1, "-"),
            ("Val Private C3/C2 Loss", r"$C_2$ on $D_1$", COLOR_C2, (0, (5, 3))),
            ("Val Private C4/C2 Loss", r"$C_3$ on $D_1$", COLOR_C3, (0, (5, 3))),
        ],
    },
    "D2": {
        "title": r"Performance on $\boldsymbol{D}_2$",
        "matched": "C2",
        "series": [
            ("Val Private C2/C3 Loss", r"$C_1$ on $D_2$", COLOR_C1, (0, (5, 3))),
            ("Val Private C3/C3 Loss", r"$C_2$ on $D_2$", COLOR_C2, "-"),
            ("Val Private C4/C3 Loss", r"$C_3$ on $D_2$", COLOR_C3, (0, (5, 3))),
        ],
    },
    "D3": {
        "title": r"Performance on $\boldsymbol{D}_3$",
        "matched": "C3",
        "series": [
            ("Val Private C2/C4 Loss", r"$C_1$ on $D_3$", COLOR_C1, (0, (5, 3))),
            ("Val Private C3/C4 Loss", r"$C_2$ on $D_3$", COLOR_C2, (0, (5, 3))),
            ("Val Private C4/C4 Loss", r"$C_3$ on $D_3$", COLOR_C3, "-"),
        ],
    },
    "Dpub": {
        "title": r"Performance on $\boldsymbol{D}_\mathrm{pub}$",
        "matched": "Cpub",
        "series": [
            ("Val Retain/C1 Loss", r"$C_\mathrm{pub}$ on $D_\mathrm{pub}$",
             COLOR_PUB, "-"),
            ("Val Retain/C2 Loss", r"$C_1$ on $D_\mathrm{pub}$",
             COLOR_C1, (0, (5, 3))),
            ("Val Retain/C3 Loss", r"$C_2$ on $D_\mathrm{pub}$",
             COLOR_C2, (0, (5, 3))),
            ("Val Retain/C4 Loss", r"$C_3$ on $D_\mathrm{pub}$",
             COLOR_C3, (0, (5, 3))),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/multi_stage_grid.png")
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


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def _draw_panel(ax, panel_name, panel, stage_data, offsets, boundaries,
                stage_labels):
    base_lw = 2.5

    handles = []
    for key, label, color, ls in panel["series"]:
        xs, ys = [], []
        for i, d in enumerate(stage_data):
            for step, val in sorted(d[key], key=lambda p: p[0]):
                xs.append(step + offsets[i])
                ys.append(val)
        line, = ax.plot(xs, ys, color=color, linewidth=base_lw, linestyle=ls,
                        label=label, dash_capstyle="butt",
                        solid_capstyle="butt")
        handles.append(line)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle=(0, (5, 3)),
                   linewidth=1.4, alpha=0.7)

    for i, lab in enumerate(stage_labels):
        center = offsets[i] + (boundaries[i] - offsets[i]) / 2
        x_min, x_max = ax.get_xlim() if False else (0, boundaries[-1])
        x_frac = (center - x_min) / (x_max - x_min)
        ax.text(x_frac, 1.01, lab,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=12, color="dimgray")

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    ax.set_title(panel["title"], fontsize=17, pad=22)
    if panel_name == "Dpub":
        ax.legend(handles=handles, fontsize=13, frameon=True,
                  loc="upper right", bbox_to_anchor=(1.0, 0.92))
    elif panel_name == "D3":
        ax.legend(handles=handles, fontsize=13, frameon=True,
                  loc="lower left")
    else:
        ax.legend(handles=handles, fontsize=13, frameon=True, loc="best")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    all_keys = sorted({key for p in PANELS.values()
                       for key, *_ in p["series"]})

    stage_data = []
    stage_lengths = []
    for s in STAGES:
        d = _load_stage(data_dir / s["csv"], all_keys)
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
    stage_labels = [s["label"] for s in STAGES]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=600)

    panel_layout = [
        (axes[0, 0], "D1"),
        (axes[0, 1], "D2"),
        (axes[1, 0], "D3"),
        (axes[1, 1], "Dpub"),
    ]
    for ax, name in panel_layout:
        _draw_panel(ax, name, PANELS[name], stage_data, offsets, boundaries,
                    stage_labels)
        ax.set_xlim(0, boundaries[-1])

    fig.tight_layout()
    # Reserve a tiny bottom/left margin, then place supx/supy text manually
    # so they sit centered against the panels (not the entire figure
    # bounding box, which is what fig.supx/yaxis use).
    fig.subplots_adjust(left=0.085, bottom=0.085, hspace=0.3, wspace=0.18)
    panel_left = min(ax.get_position().x0 for ax in axes.flat)
    panel_right = max(ax.get_position().x1 for ax in axes.flat)
    panel_bottom = min(ax.get_position().y0 for ax in axes.flat)
    panel_top = max(ax.get_position().y1 for ax in axes.flat)
    fig.text(0.5 * (panel_left + panel_right), panel_bottom - 0.05,
             "Global Training Step",
             ha="center", va="center", fontsize=18)
    fig.text(panel_left - 0.055, 0.5 * (panel_bottom + panel_top),
             "Validation Loss",
             ha="center", va="center", rotation="vertical", fontsize=18)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
