#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Qualitative C1 vs C2 generation on one sampled Alpaca prompt (530M model).
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
ALPACA_JSON=${ALPACA_JSON:-/work/scratch/data/raw/alpaca/alpaca_data.json}

SAMPLE_INDEX=${SAMPLE_INDEX:-}
SEED=${SEED:-42}
BATCH_DEVICE=${BATCH_DEVICE:-auto}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}

OUTPUT_JSON=${OUTPUT_JSON:-}

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi
if [ ! -f "$ALPACA_JSON" ]; then
    echo "Missing ALPACA_JSON: $ALPACA_JSON"
    exit 1
fi

echo "=========================================================="
echo "Qualitative C1 vs C2 (530M Alpaca)"
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Key path:        ${KEY_PATH}"
echo "  Alpaca JSON:     ${ALPACA_JSON}"
if [ -n "$SAMPLE_INDEX" ]; then
  echo "  Sample index:    ${SAMPLE_INDEX}"
else
  echo "  Sample index:    random (seed=${SEED})"
fi
echo "  Max new tokens:  ${MAX_NEW_TOKENS}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Top-p:           ${TOP_P}"
echo "  Device:          ${BATCH_DEVICE}"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$SAMPLE_INDEX" ]; then
  EXTRA_ARGS+=(--sample_index "$SAMPLE_INDEX")
fi
if [ "$DO_SAMPLE" = "1" ]; then
  EXTRA_ARGS+=(--do_sample)
fi
if [ -n "$OUTPUT_JSON" ]; then
  EXTRA_ARGS+=(--output_json "$OUTPUT_JSON")
fi

LOG_FILE="logs/qualitative_c1_c2_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:. python3 scripts/eval/qualitative_alpaca_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --alpaca_json "$ALPACA_JSON" \
  --seed "$SEED" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --device "$BATCH_DEVICE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Log file: ${LOG_FILE}"
