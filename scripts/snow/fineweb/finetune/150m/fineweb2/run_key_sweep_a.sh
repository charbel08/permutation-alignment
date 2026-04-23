#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Sweep half A: private-finetune 150M tiered pretrains on Spanish FineWeb2
# for KEY_SIZE in {0.5, 1, 2, 5}. Wraps the per-size launcher (run.sh).
#
# Runs sequentially. Set SKIP_EXISTING=1 (default) to skip runs whose
# `${OUTPUT_ROOT}/key${size}pct/final` already exists.
# ---------------------------------------------------------------------------

KEY_SIZES=${KEY_SIZES:-"0.5 1 2 5"}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}

OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key_sweep_kl${KL_TAG}}

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

WANDB_PROJECT=${WANDB_PROJECT:-finetune-sweep}
SKIP_EXISTING=${SKIP_EXISTING:-1}

echo "=========================================================="
echo "Spanish FineWeb2 finetune sweep A (150M tiered)"
echo "  Key sizes:     ${KEY_SIZES}"
echo "  KL lambda:     ${KL_LAMBDA}"
echo "  Private data:  ${PRIVATE_DATA}"
echo "  Public data:   ${PUBLIC_DATA}"
echo "  Output root:   ${OUTPUT_ROOT}"
echo "  W&B project:   ${WANDB_PROJECT}"
echo "  Skip existing: ${SKIP_EXISTING}"
echo "=========================================================="

for KEY_SIZE in $KEY_SIZES; do
    KEY_TAG=${KEY_SIZE//./p}
    BASE_CHECKPOINT="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint"
    KEY_PATH="/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json"
    OUTPUT_DIR="${OUTPUT_ROOT}/key${KEY_TAG}pct"
    RUN_NAME="finetune_150m_fineweb2_spa_key${KEY_TAG}pct_kl${KL_TAG}"

    if [ ! -d "$BASE_CHECKPOINT" ]; then
        echo "[skip] Missing BASE_CHECKPOINT for ${KEY_SIZE}%: $BASE_CHECKPOINT"
        continue
    fi
    if [ ! -f "$KEY_PATH" ]; then
        echo "[skip] Missing KEY_PATH for ${KEY_SIZE}%: $KEY_PATH"
        continue
    fi
    if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING" = "1" ]; then
        echo "[skip] Already complete: ${OUTPUT_DIR}/final"
        continue
    fi

    echo ""
    echo "=========================================================="
    echo ">>> Launching KEY_SIZE=${KEY_SIZE}%"
    echo "    Base checkpoint: ${BASE_CHECKPOINT}"
    echo "    Output dir:      ${OUTPUT_DIR}"
    echo "    Run name:        ${RUN_NAME}"
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
done

echo ""
echo "=========================================================="
echo "Spanish finetune sweep A complete."
echo "=========================================================="
