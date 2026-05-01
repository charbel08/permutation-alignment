#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-qwen_key_destruction_0p5pct_to_20pct_mmlu_h100_1gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-tiered-alignment-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
TOKENIZER_ID="${TOKENIZER_ID:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

KEY_PCTS="${KEY_PCTS:-0.005 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20}"
ATTN_RATIO="${ATTN_RATIO:-0.25}"
SEED="${SEED:-42}"

SHOTS="${SHOTS:-5}"
MAX_EXAMPLES_PER_SUBJECT="${MAX_EXAMPLES_PER_SUBJECT:--1}"
DATASET_NAME="${DATASET_NAME:-cais/mmlu}"
DTYPE="${DTYPE:-bfloat16}"

WANDB="${WANDB:-1}"
DISABLE_MATH500="${DISABLE_MATH500:-1}"

mkdir -p "$(dirname "$LOG_FILE")"

CMD=(
    python
    scripts/eval/qwen_key_destruction_ablation.py
    --model_id "$MODEL_ID"
    --key_pcts
    $KEY_PCTS
    --attn_ratio "$ATTN_RATIO"
    --seed "$SEED"
    --shots "$SHOTS"
    --max_examples_per_subject "$MAX_EXAMPLES_PER_SUBJECT"
    --dataset_name "$DATASET_NAME"
    --dtype "$DTYPE"
    --run_name "$RUN_NAME"
)

if [[ -n "$TOKENIZER_ID" ]]; then
    CMD+=(--tokenizer_id "$TOKENIZER_ID")
fi
if [[ "$DISABLE_MATH500" == "1" ]]; then
    CMD+=(--disable_math500)
else
    CMD+=(--enable_math500)
fi
if [[ "$WANDB" == "1" ]]; then
    CMD+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_NAME")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Running: PYTHONPATH=./src ${CMD[*]} $*"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH=./src "${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
