#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# LLM-as-judge C1 vs C2 for the MIXED-variant 530M instruct-tuned tiered
# model (output of run_instruction_tune_mix.sh).
#
# Phase 1: Generate C1/C2 responses on AlpacaEval (multi-GPU via torchrun).
# Phase 2: Judge with gpt-oss-120b via transformers (rank 0 only, single GPU).
#
# Pair this with run_alpaca_eval_c1_vs_c2.sh (Phase 2 of full AlpacaEval) by
# pointing JUDGE_OUTPUT_DIR=$OUTPUT_DIR to the dir produced here.
#
# Everything runs locally — no API keys, no vLLM.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}

# Mixed loss weights — must match the run_instruction_tune_mix.sh values
# whose checkpoint you want to evaluate.
W_PRIV=${W_PRIV:-0.8}
W_PUB_C2=${W_PUB_C2:-0.1}
W_PUB_C1=${W_PUB_C1:-0.1}
PRIV_TAG=${W_PRIV//./p}
PUB2_TAG=${W_PUB_C2//./p}
PUB1_TAG=${W_PUB_C1//./p}
WEIGHT_TAG=priv${PRIV_TAG}_pub2${PUB2_TAG}_pub1${PUB1_TAG}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/instruction_tune_530m_alpaca_mix_key${KEY_SIZE}pct_${WEIGHT_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/llm_judge_530m_alpaca_mix_key${KEY_SIZE}pct_${WEIGHT_TAG}}

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

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

uv pip install -q -U "triton>=3.4" kernels "transformers>=4.55"

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "LLM-as-judge C1 vs C2 — MIXED variant (530M Alpaca)"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Key path:     ${KEY_PATH}"
echo "  Loss weights: w_priv=${W_PRIV} w_pub_c2=${W_PUB_C2} w_pub_c1=${W_PUB_C1}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GPUs:         ${NGPUS}"
echo "  Judge model:  ${JUDGE_MODEL}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Max tokens:   ${MAX_NEW_TOKENS}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Top-p:        ${TOP_P}"
echo "  Do sample:    ${DO_SAMPLE}"
echo "  Difficulty:   ${DIFFICULTY_MODE} (field=${DIFFICULTY_FIELD})"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$MAX_INSTANCES")
fi
if [ "$DO_SAMPLE" = "1" ]; then
  EXTRA_ARGS+=(--do_sample)
fi

LOG_FILE="logs/llm_judge_c1_vs_c2_530m_mix_key${KEY_SIZE}pct_${WEIGHT_TAG}_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:. torchrun --standalone --nproc_per_node="$NGPUS" \
  scripts/eval/llm_judge_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --judge_model "$JUDGE_MODEL" \
  --judge_batch_size "$JUDGE_BATCH_SIZE" \
  --judge_max_tokens "$JUDGE_MAX_TOKENS" \
  --difficulty_mode "$DIFFICULTY_MODE" \
  --difficulty_field "$DIFFICULTY_FIELD" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results: ${OUTPUT_DIR}/judge_results.json"
echo "Log file:      ${LOG_FILE}"
