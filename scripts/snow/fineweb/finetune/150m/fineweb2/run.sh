#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Single private-finetune launch for Spanish FineWeb2 on a 150M tiered model.
#
# Defaults are wired to current 150M key-size checkpoints:
#   KEY_SIZE=2  -> tiered_pretrain_150m_2pct/final-checkpoint
#   KEY_SIZE=5  -> tiered_pretrain_150m_5pct/final-checkpoint
#   KEY_SIZE=10 -> tiered_pretrain_150m_10pct/final-checkpoint
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}               # one of: 2, 5, 10
KL_LAMBDA=${KL_LAMBDA:-0.1}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}

KL_TAG=${KL_LAMBDA//./p}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-finetune}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_spa_key${KEY_SIZE}pct_kl${KL_TAG}}

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

TRAIN_SAMPLES=$(python - "$PRIVATE_DATA" <<'PY'
from datasets import load_from_disk
import sys

ds = load_from_disk(sys.argv[1])
if hasattr(ds, "keys") and "train" in ds:
    print(len(ds["train"]))
else:
    print(len(ds))
PY
)

# private_finetune uses DistributedSampler(drop_last=False) + DataLoader(drop_last=True).
SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))

if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
    echo "Computed <1 step/epoch. Adjust BATCH_SIZE/NGPUS."
    exit 1
fi

if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
else
    # Default: exactly 1 epoch on Spanish private train split.
    RUN_MAX_STEPS="$STEPS_PER_EPOCH"
fi

echo "=========================================================="
echo "Private finetune (Spanish FineWeb2, 150M tiered)"
echo "  Key size:       ${KEY_SIZE}%"
echo "  KL lambda:      ${KL_LAMBDA}"
echo "  Base checkpoint:${BASE_CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Private data:   ${PRIVATE_DATA}"
echo "  Public data:    ${PUBLIC_DATA}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  GPUs:           ${NGPUS}"
echo "  Train rows:     ${TRAIN_SAMPLES}"
echo "  Steps/epoch:    ${STEPS_PER_EPOCH}"
echo "  Max steps:      ${RUN_MAX_STEPS}"
echo "=========================================================="

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune \
    --checkpoint "$BASE_CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --private_data "$PRIVATE_DATA" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --max_steps "$RUN_MAX_STEPS" \
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
    2>&1 | tee "$LOG_FILE"
