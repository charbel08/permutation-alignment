#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-qwen_key_destruction_0p5pct_to_20pct_mmlu_h100_2gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-tiered-alignment-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
TOKENIZER_ID="${TOKENIZER_ID:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NGPUS="${NGPUS:-2}"

MIN_PCT="${MIN_PCT:-0.005}"
MAX_PCT="${MAX_PCT:-0.20}"
STEP_PCT="${STEP_PCT:-0.005}"
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
    torchrun
    --standalone
    --nproc_per_node="$NGPUS"
    scripts/eval/qwen_key_destruction_ablation.py
    --model_id "$MODEL_ID"
    --min_pct "$MIN_PCT"
    --max_pct "$MAX_PCT"
    --step_pct "$STEP_PCT"
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
echo "NGPUS=${NGPUS}"
echo "Running: PYTHONPATH=./src ${CMD[*]} $*"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH=./src "${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
