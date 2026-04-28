#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Run C1 weight + public-activation magnitude analysis on the FINAL
# PRETRAIN checkpoints. Pretrain is public-only, so we drop the private
# side and only run the public-data activation pass.
# Outputs land in the same tidy tree as the finetune analyses for
# direct comparison:
#   ${OUTPUT_ROOT}/150m_pretrain/
#   ${OUTPUT_ROOT}/530m_pretrain/

KEY_SIZE=${KEY_SIZE:-5}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PUBLIC_SPLIT=${PUBLIC_SPLIT:-train}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_BATCHES=${NUM_BATCHES:-32}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-0}

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
    if [ ! -d "$PUBLIC_DATA" ]; then echo "Missing PUBLIC_DATA: $PUBLIC_DATA"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --public_data "$PUBLIC_DATA" \
        --public_split "$PUBLIC_SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --num_batches "$NUM_BATCHES" \
        --max_length "$MAX_LENGTH" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "$log_file"
}

run_pretrain_150m_cumulative_15pct() {
    # Cumulative 150M pretrain trained with three disjoint 5%-keys.
    # Largest tier (C4) corresponds to the union: 15% of weights.
    local out_dir="${OUTPUT_ROOT}/150m_pretrain_cumulative_15pct"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint"
    local key1="/work/permutation-alignment/configs/keys/150m/both/key_5pct_1.json"
    local key2="/work/permutation-alignment/configs/keys/150m/both/key_5pct_2.json"
    local key3="/work/permutation-alignment/configs/keys/150m/both/key_5pct_3.json"
    local out_json="${out_dir}/analysis_150m_pretrain_cumulative_15pct_c1_magnitudes.json"
    local log_file="logs/c1_magnitude_analysis_pretrain_150m_cumulative_15pct_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Running 150M pretrain (cumulative, largest tier = 15%) C1 magnitude analysis"
    echo "  Checkpoint: ${ckpt}"
    echo "  Keys (union): ${key1} ${key2} ${key3}"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    for k in "$key1" "$key2" "$key3"; do
        if [ ! -f "$k" ]; then echo "Missing key: $k"; exit 1; fi
    done
    if [ ! -d "$PUBLIC_DATA" ]; then echo "Missing PUBLIC_DATA: $PUBLIC_DATA"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "$key1" "$key2" "$key3" \
        --public_data "$PUBLIC_DATA" \
        --public_split "$PUBLIC_SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --num_batches "$NUM_BATCHES" \
        --max_length "$MAX_LENGTH" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
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
    if [ ! -d "$PUBLIC_DATA" ]; then echo "Missing PUBLIC_DATA: $PUBLIC_DATA"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --public_data "$PUBLIC_DATA" \
        --public_split "$PUBLIC_SPLIT" \
        --batch_size "$BATCH_SIZE" \
        --num_batches "$NUM_BATCHES" \
        --max_length "$MAX_LENGTH" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED" \
        --output_path "$out_json" \
        --plot_dir "$out_dir" 2>&1 | tee "$log_file"
}

case "$TARGET" in
    all)
        run_pretrain_150m
        run_pretrain_150m_cumulative_15pct
        run_pretrain_530m
        ;;
    150m|150m_pretrain)
        run_pretrain_150m
        ;;
    150m_cumulative|150m_cumulative_15pct|cumulative)
        run_pretrain_150m_cumulative_15pct
        ;;
    530m|530m_pretrain)
        run_pretrain_530m
        ;;
    *)
        echo "Unsupported TARGET: ${TARGET}"
        echo "Expected one of: all, 150m, 150m_cumulative, 530m"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Pretrain outputs: ${OUTPUT_ROOT}/{150m,530m}_pretrain"
