#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Run the existing LLM-as-judge AlpacaEval pipeline on every saved
# instruction-tuning checkpoint.
#
# This reuses run_llm_judge_c1_vs_c2.sh as-is so generation and judging stay
# identical to the standalone evaluation path.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_TAG:-${KL_LAMBDA//./p}}

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/instruction_tune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
EVAL_ROOT=${EVAL_ROOT:-/work/scratch/checkpoints/fineweb/evals/llm_judge_instruction_tune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_INSTANCES=${MAX_INSTANCES:-}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}

JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-4}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}
DIFFICULTY_MODE=${DIFFICULTY_MODE:-auto}
DIFFICULTY_FIELD=${DIFFICULTY_FIELD:-difficulty}

INCLUDE_FINAL=${INCLUDE_FINAL:-1}

if [ ! -d "$TRAIN_OUTPUT_DIR" ]; then
    echo "Missing TRAIN_OUTPUT_DIR: $TRAIN_OUTPUT_DIR"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

mapfile -t CHECKPOINT_NAMES < <(
    find "$TRAIN_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
)

if [ "$INCLUDE_FINAL" = "1" ] && [ -d "$TRAIN_OUTPUT_DIR/final" ]; then
    CHECKPOINT_NAMES+=("final")
fi

if [ "${#CHECKPOINT_NAMES[@]}" -eq 0 ]; then
    echo "No checkpoint-* directories found under $TRAIN_OUTPUT_DIR"
    exit 1
fi

mkdir -p "$EVAL_ROOT"

echo "=========================================================="
echo "LLM-as-judge sweep over instruction-tuning checkpoints"
echo "  Train output:  ${TRAIN_OUTPUT_DIR}"
echo "  Eval root:     ${EVAL_ROOT}"
echo "  Key path:      ${KEY_PATH}"
echo "  GPUs:          ${NGPUS}"
echo "  Judge model:   ${JUDGE_MODEL}"
echo "  Checkpoints:   ${#CHECKPOINT_NAMES[@]}"
for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    echo "    - ${checkpoint_name}"
done
echo "=========================================================="

for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    checkpoint_path="${TRAIN_OUTPUT_DIR}/${checkpoint_name}"
    eval_output_dir="${EVAL_ROOT}/${checkpoint_name}"

    echo ""
    echo ">>> Evaluating ${checkpoint_name}"

    KEY_SIZE="$KEY_SIZE" \
    KL_TAG="$KL_TAG" \
    CHECKPOINT="$checkpoint_path" \
    KEY_PATH="$KEY_PATH" \
    OUTPUT_DIR="$eval_output_dir" \
    NGPUS="$NGPUS" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    MAX_INSTANCES="$MAX_INSTANCES" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    DO_SAMPLE="$DO_SAMPLE" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    JUDGE_BATCH_SIZE="$JUDGE_BATCH_SIZE" \
    JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
    DIFFICULTY_MODE="$DIFFICULTY_MODE" \
    DIFFICULTY_FIELD="$DIFFICULTY_FIELD" \
    bash scripts/snow/fineweb/finetune/530m/alpaca/run_llm_judge_c1_vs_c2.sh
done
