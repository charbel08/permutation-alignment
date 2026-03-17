#!/usr/bin/env bash
set -euo pipefail

# Thin launcher for Qwen MMLU key-destruction ablation.
#
# The Python script handles range generation, baseline evaluation, nested
# keys, CIs, and output formatting.  This wrapper just sets environment
# defaults and forwards arguments.
#
# Quick dev run:
#   ./scripts/eval/run_qwen_key_destruction.sh --max_examples_per_subject 20
#
# Full publishable run (default):
#   ./scripts/eval/run_qwen_key_destruction.sh
#
# Custom model + range:
#   ./scripts/eval/run_qwen_key_destruction.sh \
#     --model_id Qwen/Qwen3-4B \
#     --min_pct 0.01 --max_pct 0.50 --step_pct 0.01

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHONPATH=./src python scripts/eval/mmlu_qwen_key_ablation.py "$@"