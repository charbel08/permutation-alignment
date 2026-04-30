#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Weight-only magnitude analysis for three models, each producing:
#   - weights_per_layer_ratio_heatmap.{png,pdf}
#   - weights_per_layer_ratio_heatmap_random_baseline.{png,pdf}
# plus the JSON with both stat dicts so heatmaps can be redrawn locally
# via scripts/eval/replot_magnitude_heatmap.py.
#
# Models:
#   c2k          : c2k Spanish finetune (k=20 by default; override with K_LABEL)
#   pretrain     : tiered pretrain 5%-key
#   multi_stage  : final stage of multi-stage cumulative finetune (15% union)

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}
SEED=${SEED:-42}
TARGET=${TARGET:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/permutation-alignment/outputs/c1_magnitude_three_models}

# c2k
K_LABEL=${K_LABEL:-resweep_a_k20}
C2K_CHECKPOINT=${C2K_CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_c2k_key${KEY_SIZE}pct_kl${KL_TAG}/${K_LABEL}/final}
C2K_KEY=${C2K_KEY:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

# Tiered pretrain
PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint}
PRETRAIN_KEY=${PRETRAIN_KEY:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

# Multi-stage cumulative finetune (last stage is C4 with 3 stages indexed 0..2)
MS_TAG=${MS_TAG:-perconfig}
MS_RUN_SUFFIX=${MS_RUN_SUFFIX:-}
MS_CHECKPOINT=${MS_CHECKPOINT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_multi_stage${MS_RUN_SUFFIX}_${MS_TAG}_key${KEY_SIZE}pct_kl${KL_TAG}/stage_2_C4/final}
MS_KEY1=${MS_KEY1:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct_1.json}
MS_KEY2=${MS_KEY2:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct_2.json}
MS_KEY3=${MS_KEY3:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct_3.json}

mkdir -p "$OUTPUT_ROOT"

run_analysis() {
    local label="$1"; shift
    local checkpoint="$1"; shift
    local out_json="$1"; shift
    local out_dir="$1"; shift
    # Remaining args = key paths
    local keys=("$@")
    local log_file="logs/c1_magnitude_three_models_${label}_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Magnitude analysis: ${label}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  Keys:       ${keys[*]}"
    echo "  Out JSON:   ${out_json}"
    echo "  Plot dir:   ${out_dir}"
    echo "=========================================================="

    if [ ! -d "$checkpoint" ]; then echo "Missing checkpoint: $checkpoint"; exit 1; fi
    for k in "${keys[@]}"; do
        if [ ! -f "$k" ]; then echo "Missing key: $k"; exit 1; fi
    done

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$checkpoint" \
        --key_path "${keys[@]}" \
        --weights_only \
        --seed "$SEED" \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "$log_file"
}

run_c2k() {
    local out_dir="${OUTPUT_ROOT}/c2k_${K_LABEL}_spa"
    mkdir -p "$out_dir"
    run_analysis "c2k_${K_LABEL}_spa" \
        "$C2K_CHECKPOINT" \
        "${out_dir}/analysis_c2k_${K_LABEL}_spa_c1_magnitudes.json" \
        "$out_dir" \
        "$C2K_KEY"
}

run_pretrain() {
    local out_dir="${OUTPUT_ROOT}/pretrain_150m_${KEY_SIZE}pct"
    mkdir -p "$out_dir"
    run_analysis "pretrain_150m_${KEY_SIZE}pct" \
        "$PRETRAIN_CHECKPOINT" \
        "${out_dir}/analysis_pretrain_150m_${KEY_SIZE}pct_c1_magnitudes.json" \
        "$out_dir" \
        "$PRETRAIN_KEY"
}

run_multi_stage() {
    local out_dir="${OUTPUT_ROOT}/multi_stage_final"
    mkdir -p "$out_dir"
    run_analysis "multi_stage_final" \
        "$MS_CHECKPOINT" \
        "${out_dir}/analysis_multi_stage_final_c1_magnitudes.json" \
        "$out_dir" \
        "$MS_KEY1" "$MS_KEY2" "$MS_KEY3"
}

case "$TARGET" in
    all)
        run_c2k
        run_pretrain
        run_multi_stage
        ;;
    c2k)
        run_c2k
        ;;
    pretrain)
        run_pretrain
        ;;
    multi_stage|ms)
        run_multi_stage
        ;;
    *)
        echo "Unsupported TARGET: ${TARGET}"
        echo "Expected one of: all, c2k, pretrain, multi_stage"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Outputs: ${OUTPUT_ROOT}"
echo ""
echo "To re-plot locally from the saved JSON:"
echo "  python scripts/eval/replot_magnitude_heatmap.py \\"
echo "    --input_json <path-to-json> --plot_dir <new-plot-dir>"
