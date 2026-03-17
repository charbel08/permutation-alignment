#!/usr/bin/env bash
set -euo pipefail

# Launcher for Qwen MMLU key-destruction ablation.
#
# Single GPU:
#   ./scripts/eval/run_qwen_key_destruction.sh
#
# 8 GPUs:
#   NGPUS=8 ./scripts/eval/run_qwen_key_destruction.sh
#
# Quick dev run:
#   ./scripts/eval/run_qwen_key_destruction.sh --max_examples_per_subject 20
#
# Custom model + range:
#   NGPUS=8 ./scripts/eval/run_qwen_key_destruction.sh \
#     --model_id Qwen/Qwen3-4B \
#     --min_pct 0.01 --max_pct 0.50 --step_pct 0.01

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

NGPUS="${NGPUS:-1}"

if [ "$NGPUS" -gt 1 ]; then
    echo "Launching on $NGPUS GPUs via torchrun"
    PYTHONPATH=./src torchrun --standalone --nproc_per_node="$NGPUS" \
        scripts/eval/mmlu_qwen_key_ablation.py "$@"
else
    PYTHONPATH=./src python scripts/eval/mmlu_qwen_key_ablation.py "$@"
fi