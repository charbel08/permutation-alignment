"""Re-render the finetune c2k pareto plot from cached CSV.

Same styling as c2k_pareto_replot: clean spines, no grid, project palette.
4 series (C_pub/C_K x Private/Public). Connectors and f= labels only for
f=2000 and f=100. No baseline reference line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

CSV = Path("outputs/finetune_c2k_pareto/finetune_c2k_pareto.csv")
OUT_DIR = Path("outputs/finetune_c2k_pareto")

PURPLE = "#662E7D"
TEAL = "#008080"


def main() -> None:
    df = pd.read_csv(CSV).sort_values("flops_pct_increase").reset_index(drop=True)
    xs = df["flops_pct_increase"].tolist()
    c1_priv = df["val_private_c1_loss"].tolist()
    c2_priv = df["val_private_c2_loss"].tolist()
    c1_pub = df["val_retain_c1_loss"].tolist()
    c2_pub = df["val_retain_c2_loss"].tolist()
    ks_sorted = df["K"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    keep_ks = {2000, 20}
    # Connectors span all four curves at the chosen K's.
    for x, K, *ys in zip(xs, ks_sorted, c1_priv, c2_priv, c1_pub, c2_pub):
        if int(K) not in keep_ks:
            continue
        valid = [y for y in ys if y == y]
        if len(valid) < 2:
            continue
        ax.plot([x, x], [min(valid), max(valid)],
                color="gray", lw=3.5, ls="--", alpha=0.85, zorder=1)

    line_lw = (((2.65 * 1.25 * 1.15) * 1.10) * 1.15) * 1.15 * 0.90
    h_k_priv,   = ax.plot(xs, c2_priv, "s-",  color=PURPLE,
                          linewidth=line_lw, markersize=10,
                          label=r"$\mathcal{C}_{K}$ (Private)",     zorder=3)
    h_k_pub,    = ax.plot(xs, c2_pub,  "D:",  color=PURPLE,
                          linewidth=line_lw * 0.78, markersize=8,
                          label=r"$\mathcal{C}_{K}$ (Public)",      zorder=3)
    h_pub_priv, = ax.plot(xs, c1_priv, "o-",  color=TEAL,
                          linewidth=line_lw, markersize=10,
                          label=r"$\mathcal{C}_{\mathrm{pub}}$ (Private)", zorder=3)
    h_pub_pub,  = ax.plot(xs, c1_pub,  "^:",  color=TEAL,
                          linewidth=line_lw * 0.78, markersize=8,
                          label=r"$\mathcal{C}_{\mathrm{pub}}$ (Public)",  zorder=3)

    # f= labels for the kept connectors only, placed above the topmost point
    # at each K (= C_pub Private, the top curve).
    label_placements = {
        2000: dict(xytext=(14, 8), ha="center", va="bottom"),
        20:   dict(xytext=(0, 8),  ha="center", va="bottom"),
    }
    for x, K, *ys in zip(xs, ks_sorted, c1_priv, c2_priv, c1_pub, c2_pub):
        if int(K) not in keep_ks:
            continue
        valid = [y for y in ys if y == y]
        y_top = max(valid)
        placement = label_placements[int(K)]
        ax.annotate(rf"$f={int(K)}$", (x, y_top), textcoords="offset points",
                    fontsize=15, color="black", **placement)

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

    legend_handles = [h_k_priv, h_k_pub, h_pub_priv, h_pub_pub]
    ax.legend(legend_handles, [h.get_label() for h in legend_handles],
              ncol=2, fontsize=14.5, frameon=True,
              loc="center left", bbox_to_anchor=(0.04, 0.35))

    fig.tight_layout()

    png = OUT_DIR / "finetune_c2k_pareto.png"
    pdf = OUT_DIR / "finetune_c2k_pareto.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
