#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Control run: private finetune the non-tiered 150M baseline while updating only
# the key-selected weights in the home/C1 layout. The permutation key is never
# applied during the private training loss; it only defines the trainable subset.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_from_baseline_key${KEY_SIZE}pct_no_perm_kl${KL_TAG}}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_spa_from_baseline_key${KEY_SIZE}pct_no_perm_kl${KL_TAG}}

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
TRAIN_KEYED_WITHOUT_PERM=1 \
bash scripts/snow/fineweb/finetune/150m/fineweb2/run.sh
