"""Qwen key-destruction ablation: MMLU accuracy as a function of how
much of the permutation key has been corrupted (% of key swaps zeroed).

Same project styling as the other single-panel plots: clean spines,
no grid, project palette, large fonts, figsize (9, 7.5).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/qwen/wandb_export_2026-05-01T00_12_43.130-04_00.csv")
OUT_DIR = Path("outputs/qwen")

PURPLE = "#662E7D"

X_COL = "ablation/key_pct"
Y_COL = ("qwen_key_destruction_0p5pct_to_20pct_mmlu_h100_2gpu"
         " - ablation/micro_accuracy")


def main() -> None:
    df = pd.read_csv(CSV).sort_values(X_COL).reset_index(drop=True)
    # key_pct in CSV is a fraction (0..0.2); plot as percentage 0..20.
    df = df[df[X_COL] <= 0.10].reset_index(drop=True)
    x = df[X_COL].astype(float) * 100.0
    y = df[Y_COL].astype(float) * 100.0  # accuracy as %

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90

    ax.plot(x, y, color=PURPLE, linewidth=base_lw, marker="o",
            markersize=10, markeredgecolor="white", markeredgewidth=1.5,
            zorder=4)

    ax.set_xlabel("Percent Swapped", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("MMLU Accuracy (%)", fontsize=19.5 * 1.10, labelpad=8)

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

    ax.set_xlim(-0.3, 10.3)
    ax.set_xticks([0, 2, 4, 5, 6, 8, 10])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))

    # 5% reference: vertical dashed line + curved-arrow callout.
    ax.axvline(5, color="gray", linestyle=(0, (5, 3)),
               linewidth=2.0, alpha=0.85, zorder=1)
    ax.annotate(
        "Equivalent to the 5% key\nused in main experiments",
        xy=(5, 51), xytext=(5.5, 67),
        ha="left", va="center",
        fontsize=15, fontweight="bold", color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=2.0,
                        connectionstyle="arc3,rad=-0.4",
                        shrinkA=14, shrinkB=10),
    )

    fig.tight_layout()

    out_png = OUT_DIR / "qwen_key_destruction.png"
    out_pdf = OUT_DIR / "qwen_key_destruction.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
