#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Run the magnitude key-recovery attack on the three FINETUNED models the
# magnitude analysis covers, and write attack_metrics.json next to each
# model's heatmap directory:
#   ${OUTPUT_ROOT}/c2k_${K_LABEL}_spa/attack_metrics.json
#   ${OUTPUT_ROOT}/finetune_150m_${KEY_SIZE}pct_spa/attack_metrics.json
#   ${OUTPUT_ROOT}/multi_stage_final/attack_metrics.json
#
# Models:
#   c2k          : c2k Spanish finetune (k=20 by default; override with K_LABEL)
#   finetune     : standard private finetune (5%-key, Spanish) on top of pretrain
#   multi_stage  : final stage of multi-stage cumulative finetune (15% union)
#
# Path defaults mirror run_c1_magnitude_analysis_three_models.sh so the same
# overrides work here.

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}
TARGET=${TARGET:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/permutation-alignment/outputs/c1_magnitude_three_models}

# c2k
K_LABEL=${K_LABEL:-resweep_a_k20}
C2K_CHECKPOINT=${C2K_CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_c2k_key${KEY_SIZE}pct_kl${KL_TAG}/${K_LABEL}/final}
C2K_KEY=${C2K_KEY:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

# Standard private finetune (Spanish, 5%-key)
FT_CHECKPOINT=${FT_CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct_kl${KL_TAG}/final}
FT_KEY=${FT_KEY:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

# Multi-stage cumulative finetune (random-key variant)
MS_TAG=${MS_TAG:-perconfig}
MS_RUN_SUFFIX=${MS_RUN_SUFFIX:-_random}
MS_KEY_SUFFIX=${MS_KEY_SUFFIX:-_random}
MS_CHECKPOINT=${MS_CHECKPOINT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_multi_stage${MS_RUN_SUFFIX}_${MS_TAG}_key${KEY_SIZE}pct_kl${KL_TAG}/stage_2_C4/final}
MS_KEY1=${MS_KEY1:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${MS_KEY_SUFFIX}_1.json}
MS_KEY2=${MS_KEY2:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${MS_KEY_SUFFIX}_2.json}
MS_KEY3=${MS_KEY3:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${MS_KEY_SUFFIX}_3.json}

mkdir -p "$OUTPUT_ROOT"

run_attack() {
    local label="$1"; shift
    local checkpoint="$1"; shift
    local out_dir="$1"; shift
    local keys=("$@")
    local out_json="${out_dir}/attack_metrics.json"
    local log_file="logs/attack_three_models_${label}_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Magnitude attack: ${label}"
    echo "  Checkpoint: ${checkpoint}"
    echo "  Keys:       ${keys[*]}"
    echo "  Output:     ${out_json}"
    echo "=========================================================="

    if [ ! -d "$checkpoint" ]; then echo "Missing checkpoint: $checkpoint"; exit 1; fi
    for k in "${keys[@]}"; do
        if [ ! -f "$k" ]; then echo "Missing key: $k"; exit 1; fi
    done

    mkdir -p "$out_dir"
    PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
        --checkpoint "$checkpoint" \
        --key_path "${keys[@]}" \
        --output_path "$out_json" 2>&1 | tee "$log_file"
}

run_c2k() {
    run_attack "c2k_${K_LABEL}_spa" \
        "$C2K_CHECKPOINT" \
        "${OUTPUT_ROOT}/c2k_${K_LABEL}_spa" \
        "$C2K_KEY"
}

run_finetune() {
    run_attack "finetune_150m_${KEY_SIZE}pct_spa" \
        "$FT_CHECKPOINT" \
        "${OUTPUT_ROOT}/finetune_150m_${KEY_SIZE}pct_spa" \
        "$FT_KEY"
}

run_multi_stage() {
    run_attack "multi_stage_final" \
        "$MS_CHECKPOINT" \
        "${OUTPUT_ROOT}/multi_stage_final" \
        "$MS_KEY1" "$MS_KEY2" "$MS_KEY3"
}

case "$TARGET" in
    all)
        run_c2k
        run_finetune
        run_multi_stage
        ;;
    c2k)
        run_c2k
        ;;
    finetune|ft)
        run_finetune
        ;;
    multi_stage|ms)
        run_multi_stage
        ;;
    *)
        echo "Unsupported TARGET: ${TARGET}"
        echo "Expected one of: all, c2k, finetune, multi_stage"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Attack metrics:"
echo "  ${OUTPUT_ROOT}/c2k_${K_LABEL}_spa/attack_metrics.json"
echo "  ${OUTPUT_ROOT}/finetune_150m_${KEY_SIZE}pct_spa/attack_metrics.json"
echo "  ${OUTPUT_ROOT}/multi_stage_final/attack_metrics.json"
