"""Re-render the partial-key recovery (per-module variant) plot from
the cached summary JSON, applying the project's standard styling.

Reads outputs/partial_key_recovery_per_module_5pct_150m_synbios_key5pct_kl0p1_test/partial_key_recovery_summary.json
and writes the matching .png/.pdf in the same directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SUMMARY = Path(
    "outputs/partial_key_recovery_per_module_5pct_150m_synbios_key5pct_kl0p1_test"
    "/partial_key_recovery_summary.json"
)
OUT_DIR = SUMMARY.parent

PURPLE = "#662E7D"
TEAL = "#008080"

METRIC_LABELS = {
    "top1_acc":    "Top 1 Token",
    "exact_match": "Exact Match",
}
METRIC_COLOR = {
    "top1_acc":    TEAL,
    "exact_match": PURPLE,
}


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    summaries = payload["partial_key_summaries"]
    pcts = [float(s["pct"]) for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=600)

    base_lw = 2.5

    for mk, label in METRIC_LABELS.items():
        means = [s.get(f"{mk}_mean") for s in summaries]
        stds = [s.get(f"{mk}_std") for s in summaries]
        x, y, ylo, yhi = [], [], [], []
        for pct, m, sd in zip(pcts, means, stds):
            if m is None:
                continue
            sd = 0.0 if sd is None else float(sd)
            x.append(pct); y.append(float(m))
            ylo.append(max(0.0, float(m) - sd))
            yhi.append(min(1.0, float(m) + sd))
        if not x:
            continue
        color = METRIC_COLOR[mk]
        ax.plot(x, y, marker="o", linewidth=base_lw, markersize=8,
                color=color, label=label, zorder=3)
        ax.fill_between(x, ylo, yhi, alpha=0.18, color=color, zorder=2)

    # Vertical reference: meaningful recovery starts at 90%.
    ax.axvline(90, color="gray", linestyle=(0, (5, 3)), linewidth=3.0,
               alpha=0.85, zorder=1)
    ax.annotate(
        "Meaningful increase\nstarts after 90%",
        xy=(90, 0.55), xytext=(60, 0.65),
        fontsize=16, color="gray", ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.6),
    )

    ax.set_xlabel("Partial Key (%)", fontsize=19.5 * 1.10, labelpad=8)
    ax.set_ylabel("Accuracy (%)",   fontsize=19.5 * 1.10, labelpad=8)

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

    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y*100:.0f}"))

    ax.legend(fontsize=17, frameon=True, loc="upper left")

    fig.tight_layout()

    png = OUT_DIR / "partial_key_recovery_mean_std.png"
    pdf = OUT_DIR / "partial_key_recovery_mean_std.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
