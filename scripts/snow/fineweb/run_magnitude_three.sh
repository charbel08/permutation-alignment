#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Magnitude analysis + attack on the three finetuned models:
#   1. Regular 5% fineweb2 spa finetune (KL)
#   2. 5% mix-data spa finetune (no KL, w_priv=0.8)
#   3. Multi-stage cumulative finetune, last stage (C4) — non-random keys, KL
#
# Just run:
#   bash scripts/snow/fineweb/run_magnitude_three.sh

OUTPUT_ROOT=/work/permutation-alignment/outputs/c1_magnitude_three_models
KEYS_DIR=/work/permutation-alignment/configs/keys/150m/both
CKPT_ROOT=/work/scratch/checkpoints/fineweb

FT_LABEL=finetune_150m_5pct_spa
FT_CKPT=${CKPT_ROOT}/private_finetune_150m_fineweb2_spa_key5pct_kl0p1/final
FT_KEYS=("${KEYS_DIR}/key_5pct.json")

MIX_LABEL=mix_priv0p8
MIX_CKPT=${CKPT_ROOT}/mixed_private_finetune_150m_fineweb2_spa_key5pct_priv0p8/final
MIX_KEYS=("${KEYS_DIR}/key_5pct.json")

MS_LABEL=multi_stage_final
MS_CKPT=${CKPT_ROOT}/finetune_150m_fineweb2_multi_stage_perconfig_key5pct_kl0p1/stage_2_C4/final
MS_KEYS=("${KEYS_DIR}/key_5pct_1.json" "${KEYS_DIR}/key_5pct_2.json" "${KEYS_DIR}/key_5pct_3.json")

run_one() {
    local label="$1"; shift
    local ckpt="$1"; shift
    local keys=("$@")
    local out_dir="${OUTPUT_ROOT}/${label}"
    local out_json="${out_dir}/analysis_${label}_c1_magnitudes.json"
    local attack_json="${out_dir}/attack_metrics.json"
    local ts; ts=$(date +%Y%m%d_%H%M%S)

    echo "=========================================================="
    echo "Model: ${label}"
    echo "  Checkpoint: ${ckpt}"
    echo "  Keys:       ${keys[*]}"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    for k in "${keys[@]}"; do
        if [ ! -f "$k" ]; then echo "Missing key: $k"; exit 1; fi
    done

    mkdir -p "$out_dir"

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "${keys[@]}" \
        --weights_only \
        --seed 42 \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "logs/magnitude_three_analysis_${label}_${ts}.log"

    PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
        --checkpoint "$ckpt" \
        --key_path "${keys[@]}" \
        --output_path "$attack_json" 2>&1 | tee "logs/magnitude_three_attack_${label}_${ts}.log"
}

run_one "$FT_LABEL"  "$FT_CKPT"  "${FT_KEYS[@]}"
run_one "$MIX_LABEL" "$MIX_CKPT" "${MIX_KEYS[@]}"
run_one "$MS_LABEL"  "$MS_CKPT"  "${MS_KEYS[@]}"

echo ""
echo "Done."
echo "Outputs under: ${OUTPUT_ROOT}"
