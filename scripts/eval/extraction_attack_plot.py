"""Extraction-attack memorization plot: train vs test exact-match for
baseline, tiered pretrain, and tiered fine-tune models.

Same styling as memorization_plot.py: clean spines, no grid, project
teal/purple palette + a slate accent for the non-tiered baseline. Solid
lines = train, dashed = test. X-axis is in raw training steps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/extraction_attack/wandb_export_2026-04-29T16_55_43.000-04_00.csv")
OUT_DIR = Path("outputs/extraction_attack")

PURPLE = "#662E7D"
TEAL = "#008080"
GOLD = "#B8860B"

MODELS = [
    ("Non-TLM Pretrained",                                  GOLD,
     "attack_baseline_synbios_key5pct_frac1p00"),
    (r"TLM Pretrained $\mathcal{C}_{\mathrm{pub}}$",         TEAL,
     "attack_tiered_pretrain_synbios_key5pct_frac1p00"),
    (r"TLM Finetuned $\mathcal{C}_{\mathrm{pub}}$",          PURPLE,
     "attack_tiered_synbios_key5pct_frac1p00"),
]


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV).sort_values("Step").reset_index(drop=True)
    # CSV Step column is an eval index 0..407; the actual training run
    # ended at step 3750, so rescale linearly.
    x = df["Step"].astype(float) * (3750.0 / float(df["Step"].iloc[-1]))

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90

    for label, color, prefix in MODELS:
        y_train = df[f"{prefix} - memo_train/exact_match"]
        y_test  = df[f"{prefix} - memo_test/exact_match"]
        ax.plot(x, y_train, color=color, linestyle="-",
                linewidth=base_lw, zorder=4,
                solid_capstyle="butt")
        ax.plot(x, y_test, color=color, linestyle=(0, (5, 3)),
                linewidth=base_lw, zorder=4,
                dash_capstyle="butt")

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

    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    # Custom legend with two grouped sub-legends: Model (color) and Data
    # Split (linestyle), matching the reference image.
    model_handles = [
        mlines.Line2D([], [], color=c, lw=base_lw, label=lbl)
        for lbl, c, _ in MODELS
    ]
    split_handles = [
        mlines.Line2D([], [], color="black", lw=base_lw,
                      linestyle="-", label="Train"),
        mlines.Line2D([], [], color="black", lw=base_lw,
                      linestyle=(0, (5, 3)), label="Test"),
    ]
    blank = mlines.Line2D([], [], color="none", label="")

    handles = (
        [mlines.Line2D([], [], color="none", label=r"$\bf{Model}$")]
        + model_handles
        + [mlines.Line2D([], [], color="none", label=r"$\bf{Data\ Split}$")]
        + split_handles
    )
    ax.legend(handles=handles, ncol=1, fontsize=16, frameon=True,
              loc="center right", bbox_to_anchor=(1.0, 0.5),
              handletextpad=1.0, handlelength=3.5)

    fig.tight_layout()

    out_png = OUT_DIR / "extraction_attack.png"
    out_pdf = OUT_DIR / "extraction_attack.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
