#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-qwen_key_destruction_0p5pct_to_20pct_mmlu}"
WANDB_PROJECT="${WANDB_PROJECT:-tiered-alignment-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
KEY_PCTS="${KEY_PCTS:-0.005 0.01 0.02 0.03 0.04 0.05 0.10 0.15 0.20}"
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
    --key_pcts
    $KEY_PCTS
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
