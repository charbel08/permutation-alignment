#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Sweep launcher:
#   - key sizes: 2%, 5%, 10% (150M tiered checkpoints)
#   - KL values: 3 values (default: 0.0, 0.05, 0.1)
#
# Uses 4 GPUs per run by default (override NGPUS).
# ---------------------------------------------------------------------------

KEY_SIZES=(${KEY_SIZES:-"2 5 10"})
KL_VALUES=(${KL_VALUES:-"0.0 0.05 0.1"})

NGPUS=${NGPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-tiered-alignment-private-finetune}

PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_keysize_kl}

SKIP_EXISTING=${SKIP_EXISTING:-1}

mkdir -p "$OUTPUT_ROOT"

for KEY_SIZE in "${KEY_SIZES[@]}"; do
    BASE_CHECKPOINT="/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint"
    KEY_PATH="/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json"

    if [ ! -d "$BASE_CHECKPOINT" ]; then
        echo "Missing checkpoint for key size ${KEY_SIZE}%: $BASE_CHECKPOINT"
        exit 1
    fi
    if [ ! -f "$KEY_PATH" ]; then
        echo "Missing key for key size ${KEY_SIZE}%: $KEY_PATH"
        exit 1
    fi

    for KL_LAMBDA in "${KL_VALUES[@]}"; do
        KL_TAG=${KL_LAMBDA//./p}
        OUTPUT_DIR="${OUTPUT_ROOT}/key${KEY_SIZE}pct_kl${KL_TAG}"
        RUN_NAME="finetune_150m_fineweb2_spa_key${KEY_SIZE}pct_kl${KL_TAG}"

        if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING" = "1" ]; then
            echo "Skipping existing run: ${OUTPUT_DIR}/final"
            continue
        fi

        echo "=========================================================="
        echo "Launching key=${KEY_SIZE}%  KL=${KL_LAMBDA}"
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
        EPOCHS="$EPOCHS" \
        MAX_STEPS="$MAX_STEPS" \
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
done

echo ""
echo "Sweep complete."

