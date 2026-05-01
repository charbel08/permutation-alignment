"""Memorization (exact-match) trajectory plot for the synbios finetune.

Same styling as finetune_fineweb2_spa_plot.py: clean spines, no grid,
endpoint labels with larger math glyph, project teal/purple palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/memorization/wandb_export_2026-04-29T14_52_58.161-04_00.csv")
OUT_DIR = Path("outputs/memorization")

PURPLE = "#662E7D"
TEAL = "#008080"

RUN_PREFIX = "finetune_150m_synbios_key5pct_kl0p1 - Memo"

SERIES_COL = {
    "C2": f"{RUN_PREFIX} C2/exact_match",
    "C1": f"{RUN_PREFIX} C1/exact_match",
}

PLOT_ORDER = ["C2", "C1"]

STYLE = {
    "C2": dict(color=PURPLE, linestyle="-"),
    "C1": dict(color=TEAL,   linestyle="-"),
}

MATH_NAME = {
    "C2": r"$\mathcal{C}_{\mathrm{K}}$",
    "C1": r"$\mathcal{C}_{\mathrm{pub}}$",
}


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.sort_values("Step").reset_index(drop=True)
    # Eval index 0..486 corresponds to training steps 0..4050.
    x = df["Step"].astype(float) * (4050.0 / 486.0)
    series = {k: df[col] for k, col in SERIES_COL.items()}

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90

    for key in PLOT_ORDER:
        ax.plot(x, series[key], linewidth=base_lw, zorder=4,
                dash_capstyle="butt", solid_capstyle="butt", **STYLE[key])

    # Endpoint markers: star on C_K, circle on C_pub.
    for key in PLOT_ORDER:
        y = series[key]
        if key == "C2":
            ax.plot(
                x.iloc[-1], y.iloc[-1],
                marker="*",
                markersize=((((14.8 * 1.15 * 1.10) * 1.10) * 1.15)),
                color=STYLE[key]["color"],
                markeredgecolor="white",
                markeredgewidth=(1.0 * 1.10 * 1.15),
                zorder=6,
                clip_on=False,
            )
        else:
            ax.plot(
                x.iloc[-1], y.iloc[-1],
                marker="o",
                markersize=(((5.0 * 1.10) * 1.10) * 1.15),
                color=STYLE[key]["color"],
                zorder=6,
                clip_on=False,
            )

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Exact Match", fontsize=19.5 * 1.10, labelpad=8)

    ax.tick_params(
        axis="both", which="major",
        labelsize=15.0 * 1.10,
        length=(4.0 * 1.10) * 1.15,
        width=(0.85 * 1.10) * 1.15,
    )
    ax.tick_params(axis="both", which="minor",
                   length=(2.0 * 1.10) * 1.15, width=(0.5 * 1.10) * 1.15)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth((0.95 * 1.10) * 1.15)

    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    ymin = min(float(series[k].min()) for k in PLOT_ORDER)
    ymax = max(float(series[k].max()) for k in PLOT_ORDER)
    ypad = max((ymax - ymin) * 0.12, 0.01)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    xmin = float(x.min())
    xmax = float(x.max())
    xrange_ = xmax - xmin
    ax.set_xlim(xmin - xrange_ * 0.02, xmax + xrange_ * 0.02)

    # Endpoint labels: math glyph rendered larger than (nonexistent here)
    # text. Same pattern as val_loss_traj for visual consistency.
    math_label_size = 26.0
    x_right = 0.99
    y_offsets_pts = {
        "C2": 8 * 1.10 * 1.15,
        "C1": 8 * 1.10 * 1.15,
    }

    for key in PLOT_ORDER:
        y_end = float(series[key].iloc[-1])
        dy = y_offsets_pts[key]
        ax.annotate(
            MATH_NAME[key],
            xy=(x_right, y_end),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, dy),
            textcoords="offset points",
            ha="right",
            va="bottom" if dy > 0 else "top",
            fontsize=math_label_size,
            color=STYLE[key]["color"],
            annotation_clip=False,
        )

    fig.tight_layout()

    out_png = OUT_DIR / "memorization.png"
    out_pdf = OUT_DIR / "memorization.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
