"""Validation-loss trajectory plot for finetune_150m_fineweb2_spa_key5pct_kl0p1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/finetune_fineweb2_spa/finetune_150m_fineweb2_spa_key5pct_kl0p1.csv")
OUT_DIR = Path("outputs/finetune_fineweb2_spa")

VAL_COLS = {
    "Private/C2": "Val Private/C2 Loss",
    "Private/C1": "Val Private/C1 Loss",
    "Retain/C2": "Val Retain/C2 Loss",
    "Retain/C1": "Val Retain/C1 Loss",
}
STEP_COL = "train/step"

PURPLE = "#662E7D"
TEAL = "#008080"

PLOT_ORDER = ["Private/C2", "Private/C1", "Retain/C2", "Retain/C1"]

MATH_NAME = {
    "Private/C2": r"$\mathcal{C}_{\mathrm{K}}$",
    "Private/C1": r"$\mathcal{C}_{\mathrm{pub}}$",
    "Retain/C2":  r"$\mathcal{C}_{\mathrm{K}}$",
    "Retain/C1":  r"$\mathcal{C}_{\mathrm{pub}}$",
}
TEXT_NAME = {
    "Private/C2": "on Private Data",
    "Private/C1": "on Private Data",
    "Retain/C2":  "on Public Data",
    "Retain/C1":  "on Public Data",
}


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV)
    val_any = df[list(VAL_COLS.values())].notna().any(axis=1)
    df = df.loc[val_any, [STEP_COL] + list(VAL_COLS.values())].copy()
    df = df.sort_values(STEP_COL).reset_index(drop=True)

    series = {key: df[col] for key, col in VAL_COLS.items()}
    x = df[STEP_COL].astype(float)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15)
    lw = base_lw * 1.15 * 0.90

    style = {
        "Private/C2": dict(color=PURPLE, linestyle="-", linewidth=lw, zorder=4),
        "Private/C1": dict(color=PURPLE, linestyle=(0, (5, 3)), linewidth=lw, zorder=3),
        "Retain/C2":  dict(color=TEAL, linestyle="-", linewidth=lw, zorder=4),
        "Retain/C1":  dict(color=TEAL, linestyle=(0, (5, 3)), linewidth=lw, zorder=3),
    }

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)

    for key in PLOT_ORDER:
        ax.plot(x, series[key], **style[key], dash_capstyle="butt", solid_capstyle="butt")

    for key in PLOT_ORDER:
        y = series[key]
        if key == "Private/C2":
            ax.plot(
                x.iloc[-1], y.iloc[-1],
                marker="*",
                markersize=((((14.8 * 1.15 * 1.10) * 1.10) * 1.15)),
                color=style[key]["color"],
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
                color=style[key]["color"],
                zorder=6,
                clip_on=False,
            )

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Validation Loss", fontsize=19.5 * 1.10, labelpad=8)

    ax.tick_params(
        axis="both",
        which="major",
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

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
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

    line_label_size = 19.0
    math_label_size = 23.0
    x_right = 0.99  # right-anchor: word "Data" ends here for all four labels
    y_offsets_pts = {
        "Private/C1": 8 * 1.10 * 1.15,
        "Private/C2": 16 * 1.10 * 1.15,
        "Retain/C2":  7 * 1.10 * 1.15,
        "Retain/C1": -10 * 1.10 * 1.15,
    }

    text_annos = {}
    for key in PLOT_ORDER:
        y_end = float(series[key].iloc[-1])
        dy = y_offsets_pts[key]
        text_annos[key] = ax.annotate(
            TEXT_NAME[key],
            xy=(x_right, y_end),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, dy),
            textcoords="offset points",
            ha="right",
            va="bottom" if dy > 0 else "top",
            fontsize=line_label_size,
            color=style[key]["color"],
            annotation_clip=False,
        )

    # Force a draw so we can read each text-annotation's bbox, then anchor
    # the (larger) math glyph just to its left.
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    for key in PLOT_ORDER:
        y_end = float(series[key].iloc[-1])
        dy = y_offsets_pts[key]
        text_bbox = text_annos[key].get_window_extent()
        text_left_axes_x = inv.transform((text_bbox.x0, 0))[0]
        ax.annotate(
            MATH_NAME[key],
            xy=(text_left_axes_x, y_end),
            xycoords=ax.get_yaxis_transform(),
            xytext=(6, dy),
            textcoords="offset points",
            ha="right",
            va="bottom" if dy > 0 else "top",
            fontsize=math_label_size,
            color=style[key]["color"],
            annotation_clip=False,
        )

    fig.tight_layout()

    out_png = OUT_DIR / "val_loss_traj.png"
    out_pdf = OUT_DIR / "val_loss_traj.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
