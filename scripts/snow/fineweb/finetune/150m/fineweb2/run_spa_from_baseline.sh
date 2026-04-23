#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Tiered private finetune on Spanish FineWeb2, starting from the NON-tiered
# baseline pretrain (not the tiered 5% pretrain). Uses the full tiered recipe
# (5% key, KL, keyed L2, etc.) — only the starting checkpoint differs.
# Delegates to the shared run.sh so token budget and step count stay aligned
# with the tiered sweep.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_from_baseline_key${KEY_SIZE}pct_kl${KL_TAG}}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_spa_from_baseline_key${KEY_SIZE}pct_kl${KL_TAG}}

WANDB_PROJECT=${WANDB_PROJECT:-finetune-sweep}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}

KEY_SIZE="$KEY_SIZE" \
KL_LAMBDA="$KL_LAMBDA" \
BASE_CHECKPOINT="$BASE_CHECKPOINT" \
KEY_PATH="$KEY_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_NAME="$RUN_NAME" \
TARGET_PRIVATE_TOKENS="$TARGET_PRIVATE_TOKENS" \
WANDB_PROJECT="$WANDB_PROJECT" \
bash scripts/snow/fineweb/finetune/150m/fineweb2/run.sh
