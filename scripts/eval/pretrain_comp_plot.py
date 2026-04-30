"""Pretrain C_pub validation-loss comparison: 5% Key vs Baseline.

Same styling as the other val-loss plots — clean spines, no grid, project
teal/purple palette — but with a legend in lieu of endpoint labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/pretrain_comp/wandb_export_2026-04-29T15_11_16.578-04_00.csv")
OUT_DIR = Path("outputs/pretrain_comp")

PURPLE = "#662E7D"
TEAL = "#008080"

STEP_COL = "train/step"
SERIES = [
    (r"TLM $\mathcal{C}_{\mathrm{K}}$ (5% Key Size)",
        "pretrain_150m_fineweb_5pct - val/loss_c1", TEAL),
    ("Non-TLM baseline",
        "baseline_pretrain_150m_fineweb - val/loss_c1", PURPLE),
]


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.sort_values(STEP_COL).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90

    for label, col, color in SERIES:
        sub = df[[STEP_COL, col]].dropna()
        ax.plot(sub[STEP_COL], sub[col], color=color, linewidth=base_lw,
                solid_capstyle="butt", label=label, zorder=4)

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel(r"$\mathcal{C}_{\mathrm{pub}}$ Validation Loss",
                  fontsize=19.5 * 1.10, labelpad=8)

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

    ax.set_xlim(34000, 40000)
    ax.set_xticks([34000, 35000, 36000, 37000, 38000, 39000, 40000])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    mask = (df[STEP_COL] >= 34000) & (df[STEP_COL] <= 40000)
    visible = []
    for _, col, _ in SERIES:
        visible.extend(df.loc[mask, col].dropna().tolist())
    if visible:
        ymin, ymax = min(visible), max(visible)
        ypad = max((ymax - ymin) * 0.12, 0.005)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.legend(loc="upper right", fontsize=17, frameon=True)

    # Show "baseline reaches the same loss earlier" by anchoring at the
    # midpoint of the window: pick the baseline value at x_anchor, find
    # where 5% Key first reaches that same value (always later), then
    # drop dashed verticals and a double-headed connector.
    b = df[[STEP_COL, SERIES[1][1]]].dropna().to_numpy()
    k = df[[STEP_COL, SERIES[0][1]]].dropna().to_numpy()
    x_baseline = 35000.0
    y_match = float(np.interp(x_baseline, b[:, 0], b[:, 1]))
    x_key = float(np.interp(y_match, k[::-1, 1], k[::-1, 0]))

    y_lo = ax.get_ylim()[0]
    ax.plot([x_baseline, x_baseline], [y_lo, y_match],
            color=PURPLE, linestyle=(0, (5, 3)), linewidth=4.5, zorder=2)
    ax.plot([x_key, x_key], [y_lo, y_match],
            color=TEAL, linestyle=(0, (5, 3)), linewidth=4.5, zorder=2)
    ax.plot(x_baseline, y_match, "o", color=PURPLE,
            markersize=14, markeredgecolor="white", markeredgewidth=2.8,
            zorder=5)
    ax.plot(x_key, y_match, "o", color=TEAL,
            markersize=14, markeredgecolor="white", markeredgewidth=2.8,
            zorder=5)
    ax.annotate(
        "", xy=(x_key, y_match), xytext=(x_baseline, y_match),
        arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=4.0,
                        shrinkA=11, shrinkB=11),
        zorder=4,
    )
    pct_extra = 100.0 * (x_key - x_baseline) / x_baseline
    ax.annotate(
        f"{pct_extra:.0f}% More Steps",
        xy=(0.5 * (x_baseline + x_key), y_match),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", color=PURPLE,
    )

    fig.tight_layout()

    out_png = OUT_DIR / "pretrain_comp.png"
    out_pdf = OUT_DIR / "pretrain_comp.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()