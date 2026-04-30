"""AlpacaEval win-rate trajectory plot for the instruction-tune run.

Reads the wandb-exported CSV under outputs/instr_tune/ and renders six
curves (C_pub vs C_K) x (easy/medium/hard) using the project's purple/
teal palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

CSV = Path("outputs/instr_tune/wandb_export_2026-04-29T11_43_51.461-04_00.csv")
OUT_DIR = Path("outputs/instr_tune")

PURPLE = "#662E7D"
TEAL = "#008080"

RUN_PREFIX = ("instruction_tune_530m_alpaca_key5pct_kl0p1_llm_judge - "
              "AlpacaEval")

# (color_key, difficulty) -> CSV column suffix
SERIES = {
    ("C1", "easy"):   ("Easy",   TEAL,   "-"),
    ("C1", "medium"): ("Medium", TEAL,   (0, (5, 3))),
    ("C1", "hard"):   ("Hard",   TEAL,   (0, (1, 2))),
    ("C2", "easy"):   ("Easy",   PURPLE, "-"),
    ("C2", "medium"): ("Medium", PURPLE, (0, (5, 3))),
    ("C2", "hard"):   ("Hard",   PURPLE, (0, (1, 2))),
}

PLOT_ORDER = [
    ("C1", "easy"), ("C1", "medium"), ("C1", "hard"),
    ("C2", "easy"), ("C2", "medium"), ("C2", "hard"),
]

C_LABEL = {"C1": r"$\mathcal{C}_{\mathrm{pub}}$",
           "C2": r"$\mathcal{C}_{K}$"}


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def main() -> None:
    df = pd.read_csv(CSV)
    # Eval index 0..12 corresponds to training steps 0..4500 (375/eval).
    x = df["Step"].astype(float) * 375.0

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)

    base_lw = 3.0
    handles = []
    for key in PLOT_ORDER:
        c_key, diff = key
        diff_label, color, ls = SERIES[key]
        col = f"{RUN_PREFIX}/{diff}/{c_key} Win Rate"
        y = df[col]
        label = f"{C_LABEL[c_key]} on {diff_label}"
        h, = ax.plot(x, y, color=color, linestyle=ls, linewidth=base_lw,
                     label=label, dash_capstyle="butt", solid_capstyle="butt")
        handles.append(h)

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Win Rate", fontsize=19.5 * 1.10, labelpad=8)

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

    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    # Two columns: C_pub | C_K, three rows: easy/medium/hard.
    # matplotlib fills column-major, so pass col0 top→bot, then col1 top→bot.
    legend_handles = handles  # already [C_pub easy/med/hard, C_K easy/med/hard]
    ax.legend(legend_handles, [h.get_label() for h in legend_handles],
              ncol=2, fontsize=17, frameon=True, loc="center right",
              bbox_to_anchor=(0.98, 0.5))

    fig.tight_layout()

    out_png = OUT_DIR / "instr_tune_winrate.png"
    out_pdf = OUT_DIR / "instr_tune_winrate.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
