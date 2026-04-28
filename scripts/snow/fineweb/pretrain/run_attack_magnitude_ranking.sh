#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Run the magnitude key-recovery attack on each pretrain checkpoint and
# save metrics JSON alongside the magnitude analysis outputs:
#   ${OUTPUT_ROOT}/150m_pretrain/attack_metrics.json
#   ${OUTPUT_ROOT}/150m_pretrain_cumulative_15pct_r/attack_metrics.json
#   ${OUTPUT_ROOT}/530m_pretrain/attack_metrics.json

KEY_SIZE=${KEY_SIZE:-5}

TARGET=${TARGET:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/permutation-alignment/outputs/c1_magnitude_by_dataset}

mkdir -p "$OUTPUT_ROOT"

run_attack_150m() {
    local out_dir="${OUTPUT_ROOT}/150m_pretrain"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint"
    local key="/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json"
    local out_json="${out_dir}/attack_metrics.json"
    local log_file="logs/attack_magnitude_ranking_pretrain_150m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Magnitude attack: 150M pretrain"
    echo "  Checkpoint: ${ckpt}"
    echo "  Key:        ${key}"
    echo "  Output:     ${out_json}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    if [ ! -f "$key" ]; then echo "Missing key: $key"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --output_path "$out_json" 2>&1 | tee "$log_file"
}

run_attack_150m_cumulative_15pct_r() {
    local out_dir="${OUTPUT_ROOT}/150m_pretrain_cumulative_15pct_r"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative_random/final-checkpoint"
    local key1="/work/permutation-alignment/configs/keys/150m/both/key_5pct_random_1.json"
    local key2="/work/permutation-alignment/configs/keys/150m/both/key_5pct_random_2.json"
    local key3="/work/permutation-alignment/configs/keys/150m/both/key_5pct_random_3.json"
    local out_json="${out_dir}/attack_metrics.json"
    local log_file="logs/attack_magnitude_ranking_pretrain_150m_cumulative_15pct_r_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Magnitude attack: 150M cumulative pretrain _r (largest tier = 15%)"
    echo "  Checkpoint: ${ckpt}"
    echo "  Keys:       ${key1} ${key2} ${key3}"
    echo "  Output:     ${out_json}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    for k in "$key1" "$key2" "$key3"; do
        if [ ! -f "$k" ]; then echo "Missing key: $k"; exit 1; fi
    done

    PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
        --checkpoint "$ckpt" \
        --key_path "$key1" "$key2" "$key3" \
        --output_path "$out_json" 2>&1 | tee "$log_file"
}

run_attack_530m() {
    local out_dir="${OUTPUT_ROOT}/530m_pretrain"
    mkdir -p "$out_dir"
    local ckpt="/work/scratch/checkpoints/fineweb/tiered_pretrain_530m_${KEY_SIZE}pct/final-checkpoint"
    local key="/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json"
    local out_json="${out_dir}/attack_metrics.json"
    local log_file="logs/attack_magnitude_ranking_pretrain_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================================="
    echo "Magnitude attack: 530M pretrain"
    echo "  Checkpoint: ${ckpt}"
    echo "  Key:        ${key}"
    echo "  Output:     ${out_json}"
    echo "=========================================================="

    if [ ! -d "$ckpt" ]; then echo "Missing checkpoint: $ckpt"; exit 1; fi
    if [ ! -f "$key" ]; then echo "Missing key: $key"; exit 1; fi

    PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
        --checkpoint "$ckpt" \
        --key_path "$key" \
        --output_path "$out_json" 2>&1 | tee "$log_file"
}

case "$TARGET" in
    all)
        run_attack_150m
        run_attack_150m_cumulative_15pct_r
        run_attack_530m
        ;;
    150m|150m_pretrain)
        run_attack_150m
        ;;
    cumulative|150m_cumulative|150m_cumulative_15pct_r)
        run_attack_150m_cumulative_15pct_r
        ;;
    530m|530m_pretrain)
        run_attack_530m
        ;;
    *)
        echo "Unsupported TARGET: ${TARGET}"
        echo "Expected one of: all, 150m, 150m_cumulative, 530m"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Attack metrics: ${OUTPUT_ROOT}/{150m_pretrain,150m_pretrain_cumulative_15pct_r,530m_pretrain}/attack_metrics.json"
