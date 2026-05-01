"""KL-sweep finetune trajectories: $\\mathcal{C}_K$ private val loss vs
training step, one curve per KL coefficient.

Layout mirrors the project's key-size sweep plot but uses a teal-shaded
gradient instead of purple.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

OUT_DIR = Path("outputs/kl_sweep")
STEP_COL = "train/step"

# (csv_filename, metric_column_suffix, y_axis_label, output_basename).
PLOTS = [
    ("wandb_export_2026-04-30T21_15_51.046-04_00.csv",
     "Val Private/C2 Loss",
     r"$\mathcal{C}_{K}$ Private Validation Loss",
     "kl_sweep",
     "lower left"),
    ("wandb_export_2026-04-30T21_25_43.076-04_00.csv",
     "Val Retain/C1 Loss",
     r"$\mathcal{C}_{\mathrm{pub}}$ Public Validation Loss",
     "kl_sweep_retain_c1",
     "lower left"),
    ("wandb_export_2026-04-30T21_25_55.791-04_00.csv",
     "Val Retain/C2 Loss",
     r"$\mathcal{C}_{K}$ Public Validation Loss",
     "kl_sweep_retain_c2",
     {"loc": "upper right", "bbox_to_anchor": (0.98, 0.78)}),
]

# KL value -> wandb run name suffix.
KL_RUNS = [
    (0.0,   "kl0"),
    (0.05,  "kl0p05"),
    (0.1,   "kl0p1"),
    (0.15,  "kl0p15"),
    (0.2,   "kl0p2"),
    (0.3,   "kl0p3"),
    (0.4,   "kl0p4"),
    (0.5,   "kl0p5"),
    (0.75,  "kl0p75"),
]
RUN_PREFIX = "finetune_150m_fineweb2_spa_key5pct_"

# Teal gradient: light → dark, anchored on the project's #008080 mid-tone.
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "teal_grad", ["#A8D5D5", "#008080", "#005555"]
)


def kfmt(v, _pos):
    if abs(v) >= 1000:
        val = v / 1000
        return f"{val:.1f}K" if abs(val - int(val)) > 1e-9 else f"{int(val)}K"
    return f"{int(v)}"


def _render(csv_filename: str, metric_suffix: str,
            y_label: str, out_basename: str, legend_loc) -> None:
    csv_path = OUT_DIR / csv_filename
    df = pd.read_csv(csv_path).sort_values(STEP_COL).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=600)

    base_lw = 2.4
    n = len(KL_RUNS)
    for i, (kl, suffix) in enumerate(KL_RUNS):
        col = f"{RUN_PREFIX}{suffix} - {metric_suffix}"
        sub = df[[STEP_COL, col]].dropna()
        color = TEAL_CMAP(i / (n - 1))
        label = rf"$\mathrm{{KL}}={kl:g}$"
        ax.plot(sub[STEP_COL], sub[col],
                color=color, linewidth=base_lw, solid_capstyle="butt",
                label=label, zorder=4)

    ax.set_xlabel("Step", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel(y_label, fontsize=19.5 * 1.10, labelpad=8)

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

    handles, labels = ax.get_legend_handles_labels()
    legend_kwargs = (
        {"loc": legend_loc} if isinstance(legend_loc, str) else dict(legend_loc)
    )
    ax.legend(handles, labels, ncol=2, fontsize=14, frameon=True,
              handlelength=2.5, **legend_kwargs)

    fig.tight_layout()

    out_png = OUT_DIR / f"{out_basename}.png"
    out_pdf = OUT_DIR / f"{out_basename}.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    for csv_filename, metric_suffix, y_label, out_basename, legend_loc in PLOTS:
        _render(csv_filename, metric_suffix, y_label, out_basename, legend_loc)


if __name__ == "__main__":
    main()
