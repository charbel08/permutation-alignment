#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

# ---------------------------------------------------------------------------
# Finetune the K=100 (existing resweep_a pretrain) and K=2000 (new resweep_b
# pretrain) 150M C2K checkpoints on the Spanish FineWeb2 private dataset.
# ---------------------------------------------------------------------------

KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

RUNS=(
    "k100:/work/scratch/checkpoints/fineweb/tiered_c2k_150m_5pct_resweep_a_k100/final-checkpoint"
    "k2000:/work/scratch/checkpoints/fineweb/tiered_c2k_150m_5pct_resweep_b_k2000/final-checkpoint"
)

KEY_SIZE=5
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_c2k_k100_k2000_key${KEY_SIZE}pct_kl${KL_TAG}}

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
SKIP_EXISTING=${SKIP_EXISTING:-1}

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

mkdir -p "$OUTPUT_ROOT"

echo "=========================================================="
echo "C2K private finetune (K=100, K=2000) — Spanish FineWeb2, 150M"
echo "  Key size:      ${KEY_SIZE}%"
echo "  KL lambda:     ${KL_LAMBDA}"
echo "  Private data:  ${PRIVATE_DATA}"
echo "  Public data:   ${PUBLIC_DATA}"
echo "  Output root:   ${OUTPUT_ROOT}"
echo "  Runs:          ${#RUNS[@]}"
echo "=========================================================="

for run in "${RUNS[@]}"; do
    IFS=":" read -r RUN_LABEL BASE_CHECKPOINT <<< "$run"
    if [ ! -d "$BASE_CHECKPOINT" ]; then
        echo "Missing pretrain checkpoint for ${RUN_LABEL}: $BASE_CHECKPOINT"
        exit 1
    fi
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"
    RUN_NAME="finetune_150m_fineweb2_spa_c2k_key${KEY_SIZE}pct_${RUN_LABEL}_kl${KL_TAG}"

    if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING" = "1" ]; then
        echo "Skipping existing run: ${OUTPUT_DIR}/final"
        continue
    fi

    echo "=========================================================="
    echo "Launching ${RUN_LABEL}"
    echo "  Base checkpoint: ${BASE_CHECKPOINT}"
    echo "  Output dir:      ${OUTPUT_DIR}"
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
echo "C2K K=100/K=2000 finetune sweep complete."
