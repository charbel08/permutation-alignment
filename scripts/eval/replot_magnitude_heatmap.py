#!/usr/bin/env python3
"""Re-plot the weight L2 heatmaps from a saved magnitude-analysis JSON.

The heavy work (model load, per-channel L2 computation) lives in
analyze_c1_keyed_magnitudes.py and runs on the cluster. Its JSON contains
all the data needed to redraw the heatmaps. Use this script locally to
iterate on plot styling without re-running the analysis.

Usage:
    python scripts/eval/replot_magnitude_heatmap.py \\
        --input_json outputs/.../analysis_*_c1_magnitudes.json \\
        --plot_dir outputs/.../plots_redraw

Reads `per_layer_weight_l2_stats` and `per_layer_weight_l2_baseline_stats`
from the JSON and writes:
    weights_per_layer_ratio_heatmap.{png,pdf}
    weights_per_layer_ratio_heatmap_random_baseline.{png,pdf}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

WEIGHT_COMPONENT_ORDER = [
    "attn_q_weight_rows",
    "attn_k_weight_rows",
    "attn_v_weight_rows",
    "attn_out_weight_cols",
    "mlp_fc_weight_rows",
    "mlp_fc_bias",
    "mlp_proj_weight_cols",
]


def _load_analysis_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "analyze_c1_keyed_magnitudes", here / "analyze_c1_keyed_magnitudes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_c1_keyed_magnitudes"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_json", required=True)
    p.add_argument("--plot_dir", required=True)
    args = p.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    os.makedirs(args.plot_dir, exist_ok=True)
    amod = _load_analysis_module()

    keyed_stats = data.get("per_layer_weight_l2_stats")
    baseline_stats = data.get("per_layer_weight_l2_baseline_stats")

    if keyed_stats:
        out = os.path.join(args.plot_dir, "weights_per_layer_ratio_heatmap.png")
        amod._plot_per_layer_ratio_heatmap(
            keyed_stats,
            "Per-Layer Weight L2 Ratio Heatmap",
            WEIGHT_COMPONENT_ORDER,
            out,
            ratio_key="l2_ratio_key_over_non",
            cbar_label="Keyed / Non-Keyed L2",
        )
        print(f"Wrote {out}")
    else:
        print(f"No per_layer_weight_l2_stats in {args.input_json}")

    if baseline_stats:
        out = os.path.join(args.plot_dir, "weights_per_layer_ratio_heatmap_random_baseline.png")
        amod._plot_per_layer_ratio_heatmap(
            baseline_stats,
            "Per-Layer Weight L2 Ratio Heatmap (Random Baseline)",
            WEIGHT_COMPONENT_ORDER,
            out,
            ratio_key="l2_ratio_key_over_non",
            cbar_label="Random / Rest L2",
        )
        print(f"Wrote {out}")
    else:
        print(f"No per_layer_weight_l2_baseline_stats in {args.input_json}")

    if keyed_stats and baseline_stats:
        out = os.path.join(args.plot_dir, "weights_per_layer_ratio_heatmap_combined.png")
        amod._plot_per_layer_ratio_heatmap_pair(
            keyed_stats,
            baseline_stats,
            "Real Key",
            "Random Baseline",
            WEIGHT_COMPONENT_ORDER,
            out,
            ratio_key="l2_ratio_key_over_non",
            cbar_label="Selected / Rest L2",
            family_label="Weight Family",
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
