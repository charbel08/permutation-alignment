#!/usr/bin/env python3
"""Pair two magnitude-analysis JSONs into one combined per-layer L2-ratio
heatmap, mirroring the (Real Key vs Random Baseline) layout that
``replot_magnitude_heatmap.py`` produces for a single model.

Reads ``per_layer_weight_l2_stats`` from each input JSON (left and right) and
writes a single combined PNG/PDF using the same
``_plot_per_layer_ratio_heatmap_pair`` helper, so both panels share a
colorbar and value scale.

Usage:
    python scripts/eval/replot_magnitude_heatmap_pair.py \\
        --left_json   .../single_tier/analysis_*.json \\
        --right_json  .../multi_stage/analysis_*.json \\
        --left_title  "Single-Tier KL Finetune" \\
        --right_title "Multi-Stage KL Finetune" \\
        --plot_dir    outputs/.../pair_dir \\
        --out_name    weights_per_layer_ratio_heatmap_pair.png
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
    p.add_argument("--left_json", required=True)
    p.add_argument("--right_json", required=True)
    p.add_argument("--left_title", required=True)
    p.add_argument("--right_title", required=True)
    p.add_argument("--plot_dir", required=True)
    p.add_argument("--out_name", required=True, help="Filename for the PNG (PDF written alongside)")
    args = p.parse_args()

    with open(args.left_json) as f:
        left = json.load(f)
    with open(args.right_json) as f:
        right = json.load(f)

    left_stats = left.get("per_layer_weight_l2_stats")
    right_stats = right.get("per_layer_weight_l2_stats")
    if not left_stats:
        raise SystemExit(f"No per_layer_weight_l2_stats in {args.left_json}")
    if not right_stats:
        raise SystemExit(f"No per_layer_weight_l2_stats in {args.right_json}")

    os.makedirs(args.plot_dir, exist_ok=True)
    amod = _load_analysis_module()

    out = os.path.join(args.plot_dir, args.out_name)
    amod._plot_per_layer_ratio_heatmap_pair(
        left_stats,
        right_stats,
        args.left_title,
        args.right_title,
        WEIGHT_COMPONENT_ORDER,
        out,
        ratio_key="l2_ratio_key_over_non",
        cbar_label="Keyed / Non-Keyed L2",
        family_label="Weight Family",
    )
    print(f"Wrote {out}")
    print(f"Wrote {os.path.splitext(out)[0]}.pdf")


if __name__ == "__main__":
    main()
