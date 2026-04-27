"""Final-stage-only view: for each tier T (T=1..4), plot T's matched config
applied to its own data (solid), to every tier below T (dashed), and to
every tier above T (dotted). x-axis is the last stage's training steps.

Color = tier whose config is being applied (the perspective tier).
Linestyle = relationship between data tier and config tier.

tier 1 = public English/retain at C1 (no keys); tiers 2/3/4 = private
deu/tur/spa at C2/C3/C4.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


FINAL_STAGE_CSV = "finetune_150m_multi_stage_perconfig_stage_2_C4_spa_Latn_key5pct_kl0p1.csv"


def _retain_key(cfg: int) -> str:
    return f"Val Retain/C{cfg} Loss"


def _private_key(cfg: int, tier: int) -> str:
    return f"Val Private C{cfg}/C{tier} Loss"


# Per-tier perspective. For tier T (config C_T), build curves on every other
# tier T'. Tier 1 data is public retain (uses Val Retain/Cx Loss); tiers 2..4
# use Val Private Cx/Cy Loss. Linestyle encodes the relationship: solid for
# matched (T == T'), dashed for T' below T, dotted for T' above T.
PERSPECTIVES = [
    {"tier": 1, "label": "tier 1 (eng)", "color": "tab:gray"},
    {"tier": 2, "label": "tier 2 (deu)", "color": "tab:blue"},
    {"tier": 3, "label": "tier 3 (tur)", "color": "tab:orange"},
    {"tier": 4, "label": "tier 4 (spa)", "color": "tab:green"},
]

ALL_TIERS = [1, 2, 3, 4]
TIER_NAMES = {1: "eng", 2: "deu", 3: "tur", 4: "spa"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str,
                   default="outputs/multi_stage_finetune_history")
    p.add_argument("--output", type=str,
                   default="outputs/multi_stage_finetune_history/final_stage_per_tier.png")
    return p.parse_args()


def _load(path: Path, keys: list[str]) -> dict[str, list[tuple[int, float]]]:
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


def _key_for(cfg: int, data_tier: int) -> str:
    return _retain_key(cfg) if data_tier == 1 else _private_key(cfg, data_tier)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data_dir) / FINAL_STAGE_CSV

    keys: list[str] = []
    seen: set[str] = set()
    for p in PERSPECTIVES:
        cfg = p["tier"]
        for dt in ALL_TIERS:
            k = _key_for(cfg, dt)
            if k not in seen:
                seen.add(k)
                keys.append(k)

    series_by_key = _load(csv_path, keys)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    matched_handles = []
    perspective_handles: dict[int, list] = {}

    for p in PERSPECTIVES:
        cfg = p["tier"]
        perspective_handles[cfg] = []
        for dt in ALL_TIERS:
            key = _key_for(cfg, dt)
            pts = sorted(series_by_key[key], key=lambda q: q[0])
            if not pts:
                continue
            xs = [step for step, _ in pts]
            ys = [val for _, val in pts]
            if dt == cfg:
                style = "-"
                rel = "matched"
            elif dt < cfg:
                style = "--"
                rel = f"below (tier {dt} {TIER_NAMES[dt]})"
            else:
                style = ":"
                rel = f"above (tier {dt} {TIER_NAMES[dt]})"
            label = f"C{cfg} on tier {dt} ({TIER_NAMES[dt]}) — {rel}"
            line, = ax.plot(xs, ys, color=p["color"], label=label,
                            linewidth=1.6, linestyle=style)
            if dt == cfg:
                matched_handles.append(line)
            else:
                perspective_handles[cfg].append(line)

    ax.set_xlabel("training step (final stage only)")
    ax.set_ylabel("val loss")
    ax.grid(True, alpha=0.3)

    # Global legend: matched per-tier curves, above the axes.
    global_legend = ax.legend(handles=matched_handles,
                              bbox_to_anchor=(0.5, 1.12),
                              bbox_transform=ax.transAxes,
                              loc="lower center",
                              ncol=len(matched_handles),
                              fontsize=8, title="matched (own data)",
                              title_fontsize=9)
    ax.add_artist(global_legend)

    # One legend per perspective tier (config), stacked outside the right
    # edge of the axes.
    extra_artists = [global_legend]
    n_groups = sum(1 for hs in perspective_handles.values() if hs)
    if n_groups:
        slot_h = 1.0 / n_groups
        idx = 0
        for p in PERSPECTIVES:
            handles = perspective_handles[p["tier"]]
            if not handles:
                continue
            y_top = 1.0 - idx * slot_h
            leg = ax.legend(handles=handles,
                            bbox_to_anchor=(1.02, y_top),
                            bbox_transform=ax.transAxes,
                            loc="upper left",
                            fontsize=7,
                            title=f"C{p['tier']} ({p['label']}) on others",
                            title_fontsize=8, framealpha=0.9)
            ax.add_artist(leg)
            extra_artists.append(leg)
            idx += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                bbox_extra_artists=extra_artists)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
