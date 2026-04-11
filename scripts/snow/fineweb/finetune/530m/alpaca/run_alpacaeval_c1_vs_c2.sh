#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# AlpacaEval C1 vs C2 for 530M instruct-tuned tiered model.
# Uses Gemini as judge via its native Python SDK.
# Requires GEMINI_API_KEY in environment.
# ---------------------------------------------------------------------------

GEMINI_API_KEY=${GEMINI_API_KEY:?GEMINI_API_KEY must be set}
export GEMINI_API_KEY

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/alpacaeval_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
ANNOTATORS_CONFIG=${ANNOTATORS_CONFIG:-/work/permutation-alignment/configs/alpaca_eval/annotators/gemini_pairwise.yaml}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_INSTANCES=${MAX_INSTANCES:-}

uv pip install -q google-genai alpaca-eval

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "AlpacaEval C1 vs C2 (530M Alpaca) — Gemini judge"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Key path:     ${KEY_PATH}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GPUs:         ${NGPUS}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Max tokens:   ${MAX_NEW_TOKENS}"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$MAX_INSTANCES")
fi

LOG_FILE="logs/alpacaeval_c1_vs_c2_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:./scripts/eval:. torchrun --standalone --nproc_per_node="$NGPUS" \
  scripts/eval/alpacaeval_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --annotators_config "$ANNOTATORS_CONFIG" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Log file: ${LOG_FILE}"
