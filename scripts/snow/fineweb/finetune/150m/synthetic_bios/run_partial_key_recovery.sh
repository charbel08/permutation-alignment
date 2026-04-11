#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Partial-key recovery experiment on synthetic-bios finetuned model.
#
# Goal:
#   Measure how much C2 memorization performance can be recovered when only
#   a fraction of the correct key is available.
#
# Evaluates C2 memorization (attribute-value token accuracy) at:
#   0.1, 0.2, ... 0.9, 1, 2, ... 10, 20, ... 100 (% of key swaps),
# averaged over NUM_RUNS random subsets per percentage.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
BIO_METADATA=${BIO_METADATA:-/work/scratch/data/datasets/synthetic_bios/bios_metadata.json}

EVAL_SPLIT=${EVAL_SPLIT:-test}
TARGET_ATTR=${TARGET_ATTR:-}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_BIOS=${MAX_BIOS:-}
TOP_K=${TOP_K:-"1 3 5"}

PARTIAL_KEY_PCTS=${PARTIAL_KEY_PCTS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100"}
NUM_RUNS=${NUM_RUNS:-100}
SEED=${SEED:-42}
DEVICE=${DEVICE:-auto}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/partial_key_recovery_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}_${EVAL_SPLIT}}

echo "=========================================================="
echo "Partial-Key Recovery (Synthetic Bios, 150M)"
echo "  Checkpoint:        ${CHECKPOINT}"
echo "  Key path:          ${KEY_PATH}"
echo "  Bio metadata:      ${BIO_METADATA}"
echo "  Eval split:        ${EVAL_SPLIT}"
if [ -n "$TARGET_ATTR" ]; then
    echo "  Target attr:       ${TARGET_ATTR}"
fi
if [ -n "$MAX_BIOS" ]; then
    echo "  Max bios:          ${MAX_BIOS}"
fi
echo "  Top-k:             ${TOP_K}"
echo "  Partial key pcts:  ${PARTIAL_KEY_PCTS}"
echo "  Runs per pct:      ${NUM_RUNS}"
echo "  Seed:              ${SEED}"
echo "  Device:            ${DEVICE}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "=========================================================="

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

if [ ! -f "$BIO_METADATA" ]; then
    echo "Missing BIO_METADATA: $BIO_METADATA"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="logs/partial_key_recovery_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=()
if [ -n "$TARGET_ATTR" ]; then
    EXTRA_ARGS+=(--target_attr "$TARGET_ATTR")
fi
if [ -n "$MAX_BIOS" ]; then
    EXTRA_ARGS+=(--max_bios "$MAX_BIOS")
fi

PYTHONPATH=./src:. python3 scripts/eval/partial_key_recovery_memorization.py \
    --checkpoint "$CHECKPOINT" \
    --bio_metadata "$BIO_METADATA" \
    --key_path "$KEY_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --eval_split "$EVAL_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --top_k $TOP_K \
    --partial_key_pcts $PARTIAL_KEY_PCTS \
    --num_runs "$NUM_RUNS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Summary JSON: ${OUTPUT_DIR}/partial_key_recovery_summary.json"
echo "Summary CSV:  ${OUTPUT_DIR}/partial_key_recovery_summary.csv"
echo "Runs CSV:     ${OUTPUT_DIR}/partial_key_recovery_runs.csv"
echo "Log file:     ${LOG_FILE}"
