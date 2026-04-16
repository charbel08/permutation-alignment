#!/bin/bash
set -euo pipefail

# Resume the 530M Alpaca KL=0.03 run and train for 2 extra epochs.
# Defaults can be overridden via env vars.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KEY_SIZE="${KEY_SIZE:-5}"
KL_LAMBDA="0.03"
KL_TAG="${KL_LAMBDA//./p}"

DEFAULT_RESUME="/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final"
DEFAULT_OUTPUT="/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}_extra2ep"

export KL_LAMBDA
export RESUME_FROM="${RESUME_FROM:-$DEFAULT_RESUME}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
export EXTRA_EPOCHS_ON_RESUME="${EXTRA_EPOCHS_ON_RESUME:-2}"
export RUN_NAME="${RUN_NAME:-finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}_extra2ep}"

exec bash "${SCRIPT_DIR}/run.sh"
