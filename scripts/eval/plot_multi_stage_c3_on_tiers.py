"""C3 (keys 0+1 applied) loss on each private tier's data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STAGES = [
    {"csv": "finetune_150m_multi_stage_stage_0_C2_deu_Latn_key5pct_kl0p1.csv",
     "label": "stage — tier 2 (deu)"},
    {"csv": "finetune_150m_multi_stage_stage_1_C3_tur_Latn_key5pct_kl0p1.csv",
     "label": "stage — tier 3 (tur)"},
    {"csv": "finetune_150m_multi_stage_stage_2_C4_spa_Latn_key5pct_kl0p1.csv",
     "label": "stage — tier 4 (spa)"},
]

SERIES = [
    {"key": "Val Private C3/C2 Loss", "label": "C3 on tier 2 (deu)", "color": "tab:blue"},
    {"key": "Val Private C3/C3 Loss", "label": "C3 on tier 3 (tur)", "color": "tab:orange"},
    {"key": "Val Private C3/C4 Loss", "label": "C3 on tier 4 (spa)", "color": "tab:green"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/c3_on_tiers.png")
    return p.parse_args()


def _load_stage(path: Path, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    with path.open() as f:
        for row in csv.DictReader(f):
            step_s = row.get("train/step")
            if step_s in (None, ""):
                continue
            try:
                step = int(float(step_s))
            except ValueError:
                continue
            for k in keys:
                v = row.get(k)
                if v in (None, ""):
                    continue
                try:
                    out[k].append((step, float(v)))
                except ValueError:
                    pass
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    keys = [s["key"] for s in SERIES]

    stage_data = []
    stage_lengths = []
    for s in STAGES:
        d = _load_stage(data_dir / s["csv"], keys)
        stage_data.append(d)
        max_step = 0
        for series in d.values():
            if series:
                max_step = max(max_step, max(p[0] for p in series))
        stage_lengths.append(max_step)

    offsets = [0]
    for L in stage_lengths[:-1]:
        offsets.append(offsets[-1] + L)
    boundaries = [offsets[i] + stage_lengths[i] for i in range(len(STAGES))]

    fig, ax = plt.subplots(figsize=(10, 5))
    for s in SERIES:
        xs: list[float] = []
        ys: list[float] = []
        for i, d in enumerate(stage_data):
            for step, val in sorted(d[s["key"]], key=lambda p: p[0]):
                xs.append(step + offsets[i])
                ys.append(val)
        ax.plot(xs, ys, color=s["color"], label=s["label"],
                linewidth=1.6, marker="o", markersize=3)

    for b in boundaries[:-1]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    for i, s in enumerate(STAGES):
        center = offsets[i] + stage_lengths[i] / 2
        ax.text(center, ax.get_ylim()[1], s["label"],
                ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xlabel("cumulative training step")
    ax.set_ylabel("val private loss @ C3 (keys 0+1 applied)")
    ax.set_title("Multi-stage cumulative finetune — C3 on each private tier")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
