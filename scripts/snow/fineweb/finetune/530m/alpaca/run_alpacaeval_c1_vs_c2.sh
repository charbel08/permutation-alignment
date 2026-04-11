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
#
# C1 (public tier) is used as reference.
# Reported win-rate is C2 vs C1.
#
# Gemini option:
#   Set USE_GEMINI_JUDGE=1 and GEMINI_API_KEY in environment.
#   The script will create a temp OpenAI-client config pointing to Gemini's
#   OpenAI-compatible endpoint and use a Gemini annotator config by default.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/alpacaeval_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
NGPUS=${NGPUS:-8}

DATASET_NAME=${DATASET_NAME:-tatsu-lab/alpaca_eval}
DATASET_CONFIG=${DATASET_CONFIG:-alpaca_eval}
DATASET_SPLIT=${DATASET_SPLIT:-eval}
DATASET_JSON_PATH=${DATASET_JSON_PATH:-}
MAX_INSTANCES=${MAX_INSTANCES:-}

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}
DEVICE=${DEVICE:-auto}

C1_NAME=${C1_NAME:-C1_public}
C2_NAME=${C2_NAME:-C2_keyed}

RUN_ALPACA_EVAL=${RUN_ALPACA_EVAL:-1}
ANNOTATORS_CONFIG=${ANNOTATORS_CONFIG:-}
ALPACA_EVAL_OUTPUT_PATH=${ALPACA_EVAL_OUTPUT_PATH:-${OUTPUT_DIR}/alpaca_eval}

USE_GEMINI_JUDGE=${USE_GEMINI_JUDGE:-0}
GEMINI_API_KEY=${GEMINI_API_KEY:-}
GEMINI_BASE_URL=${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}
GEMINI_ANNOTATORS_CONFIG=${GEMINI_ANNOTATORS_CONFIG:-/work/permutation-alignment/configs/alpaca_eval/annotators/gemini_pairwise.yaml}
GEMINI_CLIENT_CONFIG=

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
echo "AlpacaEval C1 vs C2 (530M Alpaca)"
echo "  Checkpoint:           ${CHECKPOINT}"
echo "  Key path:             ${KEY_PATH}"
echo "  Output dir:           ${OUTPUT_DIR}"
echo "  GPUs:                 ${NGPUS}"
echo "  Dataset:              ${DATASET_NAME}/${DATASET_CONFIG} [${DATASET_SPLIT}]"
if [ -n "$DATASET_JSON_PATH" ]; then
  echo "  Dataset JSON path:    ${DATASET_JSON_PATH}"
fi
if [ -n "$MAX_INSTANCES" ]; then
  echo "  Max instances:        ${MAX_INSTANCES}"
fi
echo "  Batch size per rank:  ${BATCH_SIZE}"
echo "  Max new tokens:       ${MAX_NEW_TOKENS}"
echo "  C1 name:              ${C1_NAME}"
echo "  C2 name:              ${C2_NAME}"
echo "  Run alpaca_eval:      ${RUN_ALPACA_EVAL}"
if [ "$RUN_ALPACA_EVAL" = "1" ]; then
  echo "  Annotators config:    ${ANNOTATORS_CONFIG}"
  echo "  Eval output path:     ${ALPACA_EVAL_OUTPUT_PATH}"
fi
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$MAX_INSTANCES")
fi
if [ -n "$DATASET_JSON_PATH" ]; then
  EXTRA_ARGS+=(--dataset_json_path "$DATASET_JSON_PATH")
fi
if [ "$DO_SAMPLE" = "1" ]; then
  EXTRA_ARGS+=(--do_sample)
fi
if [ "$RUN_ALPACA_EVAL" = "1" ]; then
  if [ "$USE_GEMINI_JUDGE" = "1" ]; then
    if [ -z "$GEMINI_API_KEY" ]; then
      echo "USE_GEMINI_JUDGE=1 requires GEMINI_API_KEY."
      exit 1
    fi
    if [ -z "$ANNOTATORS_CONFIG" ]; then
      ANNOTATORS_CONFIG="$GEMINI_ANNOTATORS_CONFIG"
    fi
    if [ ! -f "$ANNOTATORS_CONFIG" ]; then
      echo "Missing ANNOTATORS_CONFIG: $ANNOTATORS_CONFIG"
      exit 1
    fi
    GEMINI_CLIENT_CONFIG="$(mktemp /tmp/alpaca_eval_openai_client_gemini.XXXXXX.yaml)"
    cat > "$GEMINI_CLIENT_CONFIG" <<EOF
default:
  - api_key: "${GEMINI_API_KEY}"
    base_url: "${GEMINI_BASE_URL}"
EOF
    export OPENAI_CLIENT_CONFIG_PATH="$GEMINI_CLIENT_CONFIG"
    echo "  Using Gemini judge via OPENAI_CLIENT_CONFIG_PATH=${OPENAI_CLIENT_CONFIG_PATH}"
  fi
  if [ -z "$ANNOTATORS_CONFIG" ]; then
    echo "RUN_ALPACA_EVAL=1 requires ANNOTATORS_CONFIG."
    exit 1
  fi
  EXTRA_ARGS+=(--run_alpaca_eval --annotators_config "$ANNOTATORS_CONFIG" --alpaca_eval_output_path "$ALPACA_EVAL_OUTPUT_PATH")
fi

LOG_FILE="logs/alpacaeval_c1_vs_c2_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:. torchrun --standalone --nproc_per_node="$NGPUS" \
  scripts/eval/alpacaeval_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --dataset_config "$DATASET_CONFIG" \
  --dataset_split "$DATASET_SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --device "$DEVICE" \
  --c1_name "$C1_NAME" \
  --c2_name "$C2_NAME" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "C1 outputs: ${OUTPUT_DIR}/alpacaeval_c1_outputs.json"
echo "C2 outputs: ${OUTPUT_DIR}/alpacaeval_c2_outputs.json"
if [ "$RUN_ALPACA_EVAL" = "1" ]; then
  echo "AlpacaEval outputs: ${ALPACA_EVAL_OUTPUT_PATH}"
fi
echo "Log file:   ${LOG_FILE}"
if [ -n "${GEMINI_CLIENT_CONFIG}" ] && [ -f "${GEMINI_CLIENT_CONFIG}" ]; then
  rm -f "${GEMINI_CLIENT_CONFIG}"
fi
