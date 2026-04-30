#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Partial-key recovery with a hard affected-parameter budget.
#
# Each key swap is an atom with cost equal to the number of parameters touched
# by that swap. For each target budget, the evaluator draws random atom orders
# and keeps every atom that fits within the remaining parameter budget.
# The plot x-axis is realized affected keyed parameters kept (%).
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

PARAM_BUDGET_PCTS=${PARAM_BUDGET_PCTS:-"5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"}
NUM_RUNS=${NUM_RUNS:-100}
SEED=${SEED:-42}
DEVICE=${DEVICE:-auto}
NGPUS=${NGPUS:-8}

OUTPUT_DIR=${OUTPUT_DIR:-outputs/partial_key_recovery_param_budget_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}_${EVAL_SPLIT}}

echo "=========================================================="
echo "Partial-Key Recovery (parameter-budget) - Synthetic Bios, 150M"
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
echo "  Param budget pcts: ${PARAM_BUDGET_PCTS}"
echo "  Runs per pct:      ${NUM_RUNS}"
echo "  Seed:              ${SEED}"
echo "  Device:            ${DEVICE}"
echo "  GPUs (torchrun):   ${NGPUS}"
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
LOG_FILE="logs/partial_key_recovery_param_budget_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}_$(date +%Y%m%d_%H%M%S).log"

EXTRA_ARGS=()
if [ -n "$TARGET_ATTR" ]; then
    EXTRA_ARGS+=(--target_attr "$TARGET_ATTR")
fi
if [ -n "$MAX_BIOS" ]; then
    EXTRA_ARGS+=(--max_bios "$MAX_BIOS")
fi

PYTHONPATH=./src:./scripts/eval torchrun --standalone --nproc_per_node="$NGPUS" scripts/eval/partial_key_recovery_memorization_param_budget.py \
    --checkpoint "$CHECKPOINT" \
    --bio_metadata "$BIO_METADATA" \
    --key_path "$KEY_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --eval_split "$EVAL_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --param_budget_pcts $PARAM_BUDGET_PCTS \
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
