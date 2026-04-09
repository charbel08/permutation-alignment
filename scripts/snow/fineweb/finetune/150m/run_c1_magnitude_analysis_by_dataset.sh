#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# Run the per-model C1 magnitude analysis separately for each dataset/model pair
# so the per-layer weight and activation heatmaps land in distinct directories.

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}
NUM_BATCHES=${NUM_BATCHES:-32}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-0}

DATASET_TARGET=${DATASET_TARGET:-all}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/permutation-alignment/outputs/c1_magnitude_by_dataset}

mkdir -p "$OUTPUT_ROOT"

run_synbios() {
    local out_dir="${OUTPUT_ROOT}/synthetic_bios"
    mkdir -p "$out_dir"

    echo "=========================================================="
    echo "Running synthetic_bios C1 magnitude analysis"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    KEY_SIZE="$KEY_SIZE" \
    KL_TAG="$KL_TAG" \
    NUM_BATCHES="$NUM_BATCHES" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_LENGTH="$MAX_LENGTH" \
    NUM_WORKERS="$NUM_WORKERS" \
    SEED="$SEED" \
    OUTPUT_PATH="${out_dir}/analysis_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}_c1_magnitudes.json" \
    PLOT_DIR="$out_dir" \
    bash scripts/snow/fineweb/finetune/150m/synthetic_bios/run_c1_magnitude_analysis.sh
}

run_fineweb2_spa() {
    local out_dir="${OUTPUT_ROOT}/fineweb2_spa"
    mkdir -p "$out_dir"

    echo "=========================================================="
    echo "Running fineweb2_spa C1 magnitude analysis"
    echo "  Output dir: ${out_dir}"
    echo "=========================================================="

    KEY_SIZE="$KEY_SIZE" \
    KL_TAG="$KL_TAG" \
    DATA_LANG="spa_Latn" \
    NUM_BATCHES="$NUM_BATCHES" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_LENGTH="$MAX_LENGTH" \
    NUM_WORKERS="$NUM_WORKERS" \
    SEED="$SEED" \
    OUTPUT_PATH="${out_dir}/analysis_150m_fineweb2_spa_key${KEY_SIZE}pct_kl${KL_TAG}_c1_magnitudes.json" \
    PLOT_DIR="$out_dir" \
    bash scripts/snow/fineweb/finetune/150m/fineweb2/run_c1_magnitude_analysis.sh
}

case "$DATASET_TARGET" in
    all)
        run_synbios
        run_fineweb2_spa
        ;;
    synbios|synthetic_bios)
        run_synbios
        ;;
    fineweb2|fineweb2_spa|spa)
        run_fineweb2_spa
        ;;
    *)
        echo "Unsupported DATASET_TARGET: ${DATASET_TARGET}"
        echo "Expected one of: all, synbios, fineweb2_spa"
        exit 1
        ;;
esac

echo ""
echo "Done."
echo "Per-dataset outputs: ${OUTPUT_ROOT}"
