#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Qualitative C1 vs C2 prompts for 150M Spanish FineWeb2 model.
# Uses 4 fixed prompts: 2 English + 2 Spanish.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct${KEY_SUFFIX}_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}.json}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-192}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}
DEVICE=${DEVICE:-auto}
OUTPUT_JSON=${OUTPUT_JSON:-}

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

echo "=========================================================="
echo "Qualitative C1 vs C2 (150M FineWeb2 Spanish)"
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Key path:        ${KEY_PATH}"
echo "  Max new tokens:  ${MAX_NEW_TOKENS}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Top-p:           ${TOP_P}"
echo "  Device:          ${DEVICE}"
echo "=========================================================="

EXTRA_ARGS=()
if [ "$DO_SAMPLE" = "1" ]; then
    EXTRA_ARGS+=(--do_sample)
fi
if [ -n "$OUTPUT_JSON" ]; then
    EXTRA_ARGS+=(--output_json "$OUTPUT_JSON")
fi

LOG_FILE="logs/qualitative_c1_c2_150m_fineweb2_spa_key${KEY_SIZE}pct${KEY_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:. python3 scripts/eval/qualitative_fineweb2_spa_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Log file: ${LOG_FILE}"
