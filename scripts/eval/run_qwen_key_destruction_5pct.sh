#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-qwen_key_destruction_0p5pct_to_20pct_mmlu}"
WANDB_PROJECT="${WANDB_PROJECT:-tiered-alignment-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
MIN_PCT="${MIN_PCT:-0.005}"
MAX_PCT="${MAX_PCT:-0.20}"
STEP_PCT="${STEP_PCT:-0.005}"
DISABLE_MATH500="${DISABLE_MATH500:-1}"
MATH500_DATASET_NAME="${MATH500_DATASET_NAME:-HuggingFaceH4/MATH-500}"
MATH500_SPLIT="${MATH500_SPLIT:-test}"
MAX_MATH500_EXAMPLES="${MAX_MATH500_EXAMPLES:--1}"
MATH500_MAX_NEW_TOKENS="${MATH500_MAX_NEW_TOKENS:-256}"

mkdir -p "$(dirname "$LOG_FILE")"

CMD=(
    torchrun
    --standalone
    --nproc_per_node=8
    scripts/eval/qwen_key_destruction_ablation.py
    --min_pct "$MIN_PCT"
    --max_pct "$MAX_PCT"
    --step_pct "$STEP_PCT"
    --math500_dataset_name "$MATH500_DATASET_NAME"
    --math500_split "$MATH500_SPLIT"
    --max_math500_examples "$MAX_MATH500_EXAMPLES"
    --math500_max_new_tokens "$MATH500_MAX_NEW_TOKENS"
    --wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$RUN_NAME"
    --run_name "$RUN_NAME"
)

if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ "$DISABLE_MATH500" == "1" ]]; then
    CMD+=(--disable_math500)
else
    CMD+=(--enable_math500)
fi

echo "Running: ${CMD[*]} $*"
PYTHONPATH=./src "${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
