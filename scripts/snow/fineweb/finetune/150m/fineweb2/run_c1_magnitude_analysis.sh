#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# C1 analysis for 150M FineWeb2 private-finetuned model (Spanish default):
#   - keyed vs non-key weight magnitudes
#   - keyed vs non-key activation magnitudes on private/public data
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-_random}
DATA_LANG=${DATA_LANG:-spa_Latn}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/${DATA_LANG}/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}

PRIVATE_SPLIT=${PRIVATE_SPLIT:-train}
PUBLIC_SPLIT=${PUBLIC_SPLIT:-train}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_BATCHES=${NUM_BATCHES:-32}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-0}

OUTPUT_PATH=${OUTPUT_PATH:-/work/permutation-alignment/outputs/analysis_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_kl${KL_TAG}_c1_magnitudes.json}
PLOT_DIR=${PLOT_DIR:-/work/permutation-alignment/outputs}

echo "=========================================================="
echo "C1 Magnitude Analysis (150M FineWeb2 ${DATA_LANG}, C1 config)"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Private data:   ${PRIVATE_DATA} [${PRIVATE_SPLIT}]"
echo "  Public data:    ${PUBLIC_DATA} [${PUBLIC_SPLIT}]"
echo "  Batches:        ${NUM_BATCHES}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Max length:     ${MAX_LENGTH}"
echo "  Output JSON:    ${OUTPUT_PATH}"
echo "  Plot dir:       ${PLOT_DIR}"
echo "=========================================================="

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

if [ ! -d "$PRIVATE_DATA" ]; then
    echo "Missing PRIVATE_DATA: $PRIVATE_DATA"
    exit 1
fi

if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

LOG_FILE="logs/c1_magnitude_analysis_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
    --checkpoint "$CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --private_data "$PRIVATE_DATA" \
    --public_data "$PUBLIC_DATA" \
    --private_split "$PRIVATE_SPLIT" \
    --public_split "$PUBLIC_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_batches "$NUM_BATCHES" \
    --max_length "$MAX_LENGTH" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --output_path "$OUTPUT_PATH" \
    --plot_dir "$PLOT_DIR" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Summary JSON: $OUTPUT_PATH"
echo "Plots dir:    $PLOT_DIR"
echo "Log file:     $LOG_FILE"
