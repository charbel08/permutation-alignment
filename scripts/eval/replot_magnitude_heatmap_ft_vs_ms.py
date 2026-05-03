#!/usr/bin/env python3
"""Pair the single-tier KL finetune and the multi-stage KL finetune into one
combined per-layer L2-ratio heatmap, mirroring the (Real Key vs Random
Baseline) layout that ``replot_magnitude_heatmap.py`` produces for a single
model.

Reads ``per_layer_weight_l2_stats`` from each input JSON (left = single-tier,
right = multi-stage) and writes a single combined PNG/PDF using the same
``_plot_per_layer_ratio_heatmap_pair`` helper, so both panels share a
colorbar and value scale.

Usage:
    python scripts/eval/replot_magnitude_heatmap_ft_vs_ms.py \\
        --ft_json   outputs/c1_magnitude_three_models/finetune_150m_5pct_spa/analysis_finetune_150m_5pct_spa_c1_magnitudes.json \\
        --ms_json   outputs/c1_magnitude_three_models/multi_stage_final/analysis_multi_stage_final_c1_magnitudes.json \\
        --plot_dir  outputs/c1_magnitude_three_models/ft_vs_multi_stage
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
    p.add_argument("--ft_json", required=True, help="Single-tier KL finetune analysis JSON")
    p.add_argument("--ms_json", required=True, help="Multi-stage KL finetune analysis JSON")
    p.add_argument("--plot_dir", required=True)
    p.add_argument("--ft_title", default="Single-Tier KL Finetune")
    p.add_argument("--ms_title", default="Multi-Stage KL Finetune")
    p.add_argument(
        "--out_name",
        default="weights_per_layer_ratio_heatmap_ft_vs_multi_stage.png",
    )
    args = p.parse_args()

    with open(args.ft_json) as f:
        ft = json.load(f)
    with open(args.ms_json) as f:
        ms = json.load(f)

    ft_stats = ft.get("per_layer_weight_l2_stats")
    ms_stats = ms.get("per_layer_weight_l2_stats")
    if not ft_stats:
        raise SystemExit(f"No per_layer_weight_l2_stats in {args.ft_json}")
    if not ms_stats:
        raise SystemExit(f"No per_layer_weight_l2_stats in {args.ms_json}")

    os.makedirs(args.plot_dir, exist_ok=True)
    amod = _load_analysis_module()

    out = os.path.join(args.plot_dir, args.out_name)
    amod._plot_per_layer_ratio_heatmap_pair(
        ft_stats,
        ms_stats,
        args.ft_title,
        args.ms_title,
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
