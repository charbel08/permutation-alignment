#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# LLM-as-judge C1 vs C2 for 530M instruct-tuned tiered model.
#
# Phase 1: Generate C1/C2 responses on AlpacaEval prompts (multi-GPU).
# Phase 2: Pairwise judge with gpt-oss-120b via vLLM (single H100).
#
# Position-debiased: randomly swaps A/B ordering per example.
# No external API needed — everything runs locally.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/llm_judge_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_INSTANCES=${MAX_INSTANCES:-}

JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}

uv pip install -q vllm

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
echo "LLM-as-judge C1 vs C2 (530M Alpaca)"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Key path:     ${KEY_PATH}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GPUs:         ${NGPUS}"
echo "  Judge model:  ${JUDGE_MODEL}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Max tokens:   ${MAX_NEW_TOKENS}"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$MAX_INSTANCES")
fi

LOG_FILE="logs/llm_judge_c1_vs_c2_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:. torchrun --standalone --nproc_per_node="$NGPUS" \
  scripts/eval/llm_judge_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --judge_model "$JUDGE_MODEL" \
  --judge_max_tokens "$JUDGE_MAX_TOKENS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results: ${OUTPUT_DIR}/judge_results.json"
echo "Log file:      ${LOG_FILE}"
