#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-qwen_key_destruction_5pct}"
WANDB_PROJECT="${WANDB_PROJECT:-tiered-alignment-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

CMD=(
    torchrun
    --standalone
    --nproc_per_node=8
    scripts/eval/mmlu_qwen_key_ablation.py
    --wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$RUN_NAME"
    --run_name "$RUN_NAME"
)

if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb_entity "$WANDB_ENTITY")
fi

echo "Running: ${CMD[*]} $*"
PYTHONPATH=./src "${CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
