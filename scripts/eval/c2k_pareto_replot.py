"""Re-render the c2k pretrain pareto plot from the cached CSV.

Reads outputs/c2k_pareto_public_gap/c2k_pareto.csv (already pulled from
W&B) and applies the project's standard styling: clean spines, no grid,
teal/purple palette, fontsize match.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

CSV = Path("outputs/c2k_pareto_resweep/c2k_pareto.csv")
OUT_DIR = Path("outputs/c2k_pareto_resweep")
BASELINE_COL = "baseline_val_loss_c1"

PURPLE = "#662E7D"
TEAL = "#008080"


def main() -> None:
    df = pd.read_csv(CSV).sort_values("flops_pct_increase").reset_index(drop=True)
    base_loss = float(df[BASELINE_COL].iloc[0])
    xs = df["flops_pct_increase"].tolist()
    c1 = df["val_loss_c1"].tolist()
    c2 = df["val_loss_c2"].tolist()
    ks_sorted = df["K"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    keep_ks = {2000, 20}
    for x, y1, y2, K in zip(xs, c1, c2, ks_sorted):
        if int(K) not in keep_ks or y2 != y2:
            continue
        ax.plot([x, x], [y1, y2], color="gray", lw=3.5, ls="--",
                alpha=0.85, zorder=1)

    line_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90
    ax.plot(xs, c1, "o-", color=TEAL, linewidth=line_lw, markersize=10,
            label=r"$\mathcal{C}_{\mathrm{pub}}$", zorder=3)
    ax.plot(xs, c2, "s-", color=PURPLE, linewidth=line_lw, markersize=10,
            label=r"$\mathcal{C}_{K}$", zorder=3)

    label_placements = {
        2000: dict(xytext=(10, -2), ha="left", va="center"),
        20:   dict(xytext=(0, 8),   ha="center", va="bottom"),
    }
    # Step gap (N - X), where X is the step at which C_pub first reached
    # C_K's final loss and N is the total pretraining length, expressed
    # as % of total pretraining compute. Computed offline from the runs'
    # val/loss_c1 and val/loss_c2 histories (N=45600 for both Ks):
    #   K=2000: X=28959 → (45600-28959)/45600 = 36.5%
    #   K=20:   X=41255 → (45600-41255)/45600 = 9.5%
    pct_more_steps = {2000: 36.5, 20: 9.5}
    # Anchor the right-whitespace text labels in (data) coordinates.
    text_anchors = {
        2000: (1.5, 3.185),
        20:   (15.0, 3.158),
    }

    for x, y1, y2, K in zip(xs, c1, c2, ks_sorted):
        if int(K) not in keep_ks:
            continue
        y_top = y1 if (y2 != y2) else max(y1, y2)
        placement = label_placements[int(K)]
        ax.annotate(rf"$f={int(K)}$", (x, y_top), textcoords="offset points",
                    fontsize=15, color="black", **placement)
        y_mid = 0.5 * (y1 + y2)
        tx, ty = text_anchors[int(K)]
        ax.annotate(
            f"{pct_more_steps[int(K)]:.1f}% of pretraining compute",
            xy=(x, y_mid),
            xytext=(tx, ty),
            ha="center", va="center",
            fontsize=15, fontweight="bold", color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=2.0,
                            connectionstyle="arc3,rad=-0.25",
                            shrinkA=4, shrinkB=4),
        )

    ax.set_xscale("log")

    def _decade(v, _pos):
        if v >= 1:
            return f"{int(v)}"
        return f"{v:g}"

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_decade))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("FLOPs % Increase vs Non-Tiered Baseline",
                  fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Validation Loss", fontsize=19.5 * 1.10, labelpad=8)

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

    ax.legend(ncol=2, fontsize=22, frameon=True, loc="best")
    fig.tight_layout()

    png = OUT_DIR / "c2k_pareto.png"
    pdf = OUT_DIR / "c2k_pareto.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
