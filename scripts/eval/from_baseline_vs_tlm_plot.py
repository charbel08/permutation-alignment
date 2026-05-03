"""TLM vs Non-Tiered Baseline finetune comparison: single panel showing
both C_K and C_pub validation losses on public data.

Both runs use the same key, KL recipe, and trainable subset; the only
difference is whether the starting checkpoint is the tiered pretrain
(TLM) or the non-tiered baseline pretrain.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

OUT_DIR = Path("outputs/from_baseline_vs_tlm")
TLM_CSV = OUT_DIR / "tlm.csv"
BASE_CSV = OUT_DIR / "from_baseline.csv"

PURPLE = "#662E7D"
TEAL = "#008080"

STEP_COL = "train/step"

# (csv_column, dataset_label, linestyle)
SERIES = [
    ("Val Retain/C2 Loss", r"$\mathcal{C}_{K}$",            "-"),
    ("Val Retain/C1 Loss", r"$\mathcal{C}_{\mathrm{pub}}$",  (0, (5, 3))),
]
# (df_attr_name, model_label, color)
RUNS = [
    ("tlm", "TLM",                 PURPLE),
    ("base", "Non-Tiered Baseline", TEAL),
]


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    dfs = {
        "tlm":  pd.read_csv(TLM_CSV).sort_values(STEP_COL).reset_index(drop=True),
        "base": pd.read_csv(BASE_CSV).sort_values(STEP_COL).reset_index(drop=True),
    }

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)
    base_lw = 2.6

    for run_key, _model_label, color in RUNS:
        df = dfs[run_key]
        for col, _ds_label, ls in SERIES:
            sub = df[[STEP_COL, col]].dropna()
            ax.plot(sub[STEP_COL], sub[col], color=color, linewidth=base_lw,
                    linestyle=ls, dash_capstyle="butt", solid_capstyle="butt",
                    zorder=4)

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Public Data Validation Loss",
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

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    handles = [
        Line2D([0], [0], color=PURPLE, lw=base_lw, linestyle="-",
               label=r"TLM, $\mathcal{C}_{K}$"),
        Line2D([0], [0], color=PURPLE, lw=base_lw, linestyle=(0, (5, 3)),
               label=r"TLM, $\mathcal{C}_{\mathrm{pub}}$"),
        Line2D([0], [0], color=TEAL, lw=base_lw, linestyle="-",
               label=r"Non-Tiered Baseline, $\mathcal{C}_{K}$"),
        Line2D([0], [0], color=TEAL, lw=base_lw, linestyle=(0, (5, 3)),
               label=r"Non-Tiered Baseline, $\mathcal{C}_{\mathrm{pub}}$"),
    ]
    ax.legend(handles=handles, fontsize=13, frameon=True,
              loc="upper right", handlelength=2.8)

    fig.tight_layout()

    out_png = OUT_DIR / "from_baseline_vs_tlm.png"
    out_pdf = OUT_DIR / "from_baseline_vs_tlm.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
