#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Finetune the k=1000 C2K (resweep_b) pretrain on the Spanish FineWeb2 private
# dataset. Single-run variant of run_c2k_resweep.sh.
# ---------------------------------------------------------------------------

KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

KEY_SIZE=${KEY_SIZE:-5}
RUN_LABEL=${RUN_LABEL:-k1000}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_c2k_150m_5pct_resweep_b_k1000/final-checkpoint}

KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_c2k_key${KEY_SIZE}pct_kl${KL_TAG}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_LABEL}}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_spa_c2k_key${KEY_SIZE}pct_${RUN_LABEL}_kl${KL_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-finetune-c2k}

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi
if [ ! -d "$PRIVATE_DATA" ]; then
    echo "Missing PRIVATE_DATA: $PRIVATE_DATA"
    exit 1
fi
if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "C2K k=1000 private finetune (Spanish FineWeb2, 150M)"
echo "  Base checkpoint: ${BASE_CHECKPOINT}"
echo "  Key path:        ${KEY_PATH}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Run name:        ${RUN_NAME}"
echo "  KL lambda:       ${KL_LAMBDA}"
echo "=========================================================="

KEY_SIZE="$KEY_SIZE" \
KL_LAMBDA="$KL_LAMBDA" \
BASE_CHECKPOINT="$BASE_CHECKPOINT" \
KEY_PATH="$KEY_PATH" \
PRIVATE_DATA="$PRIVATE_DATA" \
PUBLIC_DATA="$PUBLIC_DATA" \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_NAME="$RUN_NAME" \
NGPUS="$NGPUS" \
BATCH_SIZE="$BATCH_SIZE" \
LR="$LR" \
MIN_LR="$MIN_LR" \
MAX_STEPS="$MAX_STEPS" \
TARGET_PRIVATE_TOKENS="$TARGET_PRIVATE_TOKENS" \
CONTEXT_SIZE="$CONTEXT_SIZE" \
WARMUP_STEPS="$WARMUP_STEPS" \
KEYED_L2_LAMBDA="$KEYED_L2_LAMBDA" \
EVAL_INTERVAL="$EVAL_INTERVAL" \
EVAL_STEPS="$EVAL_STEPS" \
LOG_INTERVAL="$LOG_INTERVAL" \
SAVE_INTERVAL="$SAVE_INTERVAL" \
NUM_WORKERS="$NUM_WORKERS" \
WANDB_PROJECT="$WANDB_PROJECT" \
bash scripts/snow/fineweb/finetune/150m/fineweb2/run.sh

echo ""
echo "C2K k=1000 finetune complete."
