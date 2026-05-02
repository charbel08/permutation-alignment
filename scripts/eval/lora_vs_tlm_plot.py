"""LoRA 1% vs TLM 5% private finetune comparison.

Same single-panel styling as no_tlm_baseline_plot / pretrain_comp:
clean spines, no grid, project teal/purple palette, legend in lieu of
endpoint labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/lora/wandb_export_2026-05-02T12_28_02.729-04_00.csv")
OUT_DIR = Path("outputs/lora")

PURPLE = "#662E7D"
TEAL = "#008080"

STEP_COL = "train/step"
SERIES = [
    (r"TLM $\mathcal{C}_{\mathrm{K}}$ (5% Key Size)",
     "finetune_150m_fineweb2_spa_key5pct_kl0p1 - Val Private/C2 Loss",
     PURPLE),
    (r"LoRA (1% of param)",
     "lora_private_finetune_150m_fineweb2_spa_key1pct_lr1e5 - Val Private/C2 Loss",
     TEAL),
]


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV).sort_values(STEP_COL).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    base_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90

    for label, col, color in SERIES:
        sub = df[[STEP_COL, col]].dropna()
        ax.plot(sub[STEP_COL], sub[col], color=color, linewidth=base_lw,
                solid_capstyle="butt", label=label, zorder=4)

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Private Data Validation Loss",
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

    ax.legend(loc="upper right", fontsize=17, frameon=True)

    fig.tight_layout()

    out_png = OUT_DIR / "lora_vs_tlm.png"
    out_pdf = OUT_DIR / "lora_vs_tlm.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
