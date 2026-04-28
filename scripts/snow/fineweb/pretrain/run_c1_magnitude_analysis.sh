#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Run C1 weight-magnitude analysis on the FINAL PRETRAIN checkpoints.
# Pretrain is public-only, so this is weights-only (no private/public
# activation comparison). Outputs land in the same tidy tree as the
# finetune analyses for direct comparison:
#   ${OUTPUT_ROOT}/150m_pretrain/
#   ${OUTPUT_ROOT}/530m_pretrain/

KEY_SIZE=${KEY_SIZE:-5}

TARGET=${TARGET:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/permutation-alignment/outputs/c1_magnitude_by_dataset}

mkdir -p "$OUTPUT_ROOT"

run_pretrain_150m() {
    local out_dir="${OUTPUT_ROOT}/150m_pretrain"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint"
    local key="/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json"
    local out_json="${out_dir}/analysis_150m_pretrain_key${KEY_SIZE}pct_c1_magnitudes.json"
    local log_file="logs/c1_magnitude_analysis_pretrain_150m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Running 150M pretrain C1 magnitude analysis"
    echo "  Checkpoint: ${ckpt}"
    echo "  Key:        ${key}"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    if [ ! -f "$key" ]; then echo "Missing key: $key"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --weights_only \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "$log_file"
}

run_pretrain_530m() {
    local out_dir="${OUTPUT_ROOT}/530m_pretrain"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_530m_${KEY_SIZE}pct/final-checkpoint"
    local key="/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json"
    local out_json="${out_dir}/analysis_530m_pretrain_key${KEY_SIZE}pct_c1_magnitudes.json"
    local log_file="logs/c1_magnitude_analysis_pretrain_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Running 530M pretrain C1 magnitude analysis"
    echo "  Checkpoint: ${ckpt}"
    echo "  Key:        ${key}"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    if [ ! -f "$key" ]; then echo "Missing key: $key"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --weights_only \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "$log_file"
}

case "$TARGET" in
    all)
        run_pretrain_150m
        run_pretrain_530m
        ;;
    150m|150m_pretrain)
        run_pretrain_150m
        ;;
    530m|530m_pretrain)
        run_pretrain_530m
        ;;
    *)
        echo "Unsupported TARGET: ${TARGET}"
        echo "Expected one of: all, 150m, 530m"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Pretrain outputs: ${OUTPUT_ROOT}/{150m,530m}_pretrain"
