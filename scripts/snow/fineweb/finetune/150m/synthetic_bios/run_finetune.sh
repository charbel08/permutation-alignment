#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Private finetune on synthetic bios (150M tiered model).
#
# The synthetic bios dataset is small (8,640 train samples, 128 tokens each)
# so we use fewer GPUs and smaller batches to avoid exhausting the data
# too quickly per epoch.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_LAMBDA=${KL_LAMBDA:-0.1}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/synthetic_bios/tokenized}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
BIO_METADATA=${BIO_METADATA:-/work/scratch/data/datasets/synthetic_bios/bios_metadata.json}

KL_TAG=${KL_LAMBDA//./p}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/private_finetune_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-3e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-4050}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-finetune}
RUN_NAME=${RUN_NAME:-finetune_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}}
RESUME_FROM=${RESUME_FROM:-}

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
    echo "Run generate_data.sh first."
    exit 1
fi

if [ -n "$RESUME_FROM" ] && [ ! -f "$RESUME_FROM/training_state.pt" ]; then
    echo "Missing training_state.pt in RESUME_FROM: $RESUME_FROM"
    exit 1
fi

echo "=========================================================="
echo "Private finetune (Synthetic Bios, 150M tiered)"
echo "  Key size:       ${KEY_SIZE}%"
echo "  KL lambda:      ${KL_LAMBDA}"
echo "  Base checkpoint:${BASE_CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Private data:   ${PRIVATE_DATA}"
echo "  Public data:    ${PUBLIC_DATA}"
echo "  Output dir:     ${OUTPUT_DIR}"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:    ${RESUME_FROM}"
fi
echo "  GPUs:           ${NGPUS}"
echo "  Max steps:      ${MAX_STEPS}"
echo "=========================================================="

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
EXTRA_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    EXTRA_ARGS+=(--resume_from "$RESUME_FROM")
fi

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune \
    --checkpoint "$BASE_CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --private_data "$PRIVATE_DATA" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --kl_lambda "$KL_LAMBDA" \
    --keyed_l2_lambda "$KEYED_L2_LAMBDA" \
    --max_grad_norm 1.0 \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_steps "$EVAL_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --bio_metadata "$BIO_METADATA" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
