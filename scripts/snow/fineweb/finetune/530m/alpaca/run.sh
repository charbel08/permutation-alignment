#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# C2 instruction tuning on Stanford Alpaca while preserving C1 behavior.
#
# Uses tiered private_finetune objective:
#   L_ft = (1-λ) * L_instruction(C2) + λ * KL(C1_current || C1_frozen)
#
# Expects PRIVATE_DATA prepared via:
#   scripts/snow/data/prepare_alpaca.sh
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}  # one of: 2, 5, 10
KL_LAMBDA=${KL_LAMBDA:-0.1}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_530m_${KEY_SIZE}pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/alpaca/tokenized_gpt2_1024}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}

KL_TAG=${KL_LAMBDA//./p}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
EPOCHS=${EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-}
EXTRA_EPOCHS_ON_RESUME=${EXTRA_EPOCHS_ON_RESUME:-0}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-finetune}
RUN_NAME=${RUN_NAME:-finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
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
    echo "Run scripts/snow/data/prepare_alpaca.sh first."
    exit 1
fi

if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

if [ -n "$RESUME_FROM" ] && [ ! -f "$RESUME_FROM/training_state.pt" ]; then
    echo "Missing training_state.pt in RESUME_FROM: $RESUME_FROM"
    exit 1
fi

TRAIN_SAMPLES=$(python3 - "$PRIVATE_DATA" <<'PY'
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
TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * CONTEXT_SIZE ))

if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
    echo "Computed <1 step/epoch. Adjust BATCH_SIZE/NGPUS."
    exit 1
fi

if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
elif [ -n "$RESUME_FROM" ] && [ "$EXTRA_EPOCHS_ON_RESUME" -gt 0 ]; then
    RESUME_STEP=$(python3 - "$RESUME_FROM/training_state.pt" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu")
print(int(state.get("global_step", 0)))
PY
)
    EXTRA_STEPS=$(( EXTRA_EPOCHS_ON_RESUME * STEPS_PER_EPOCH ))
    RUN_MAX_STEPS=$(( RESUME_STEP + EXTRA_STEPS ))
else
    RUN_MAX_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
fi

TARGET_PRIVATE_TOKENS=$(( RUN_MAX_STEPS * TOKENS_PER_STEP ))

if [ -n "$RESUME_FROM" ]; then
    RESUME_STEP=$(python3 - "$RESUME_FROM/training_state.pt" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu")
print(int(state.get("global_step", 0)))
PY
)
    if [ "$RUN_MAX_STEPS" -le "$RESUME_STEP" ]; then
        echo "RUN_MAX_STEPS (${RUN_MAX_STEPS}) is <= resumed global_step (${RESUME_STEP})."
        echo "Set MAX_STEPS larger than ${RESUME_STEP}, or set EXTRA_EPOCHS_ON_RESUME > 0."
        exit 1
    fi
fi

echo "=========================================================="
echo "Private finetune (Alpaca instructions, 530M tiered)"
echo "  Key size:       ${KEY_SIZE}%"
echo "  KL lambda:      ${KL_LAMBDA}"
echo "  Base checkpoint:${BASE_CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Private data:   ${PRIVATE_DATA}"
echo "  Public data:    ${PUBLIC_DATA}"
echo "  Output dir:     ${OUTPUT_DIR}"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:    ${RESUME_FROM}"
    echo "  Resume step:    ${RESUME_STEP}"
fi
echo "  GPUs:           ${NGPUS}"
echo "  Context size:   ${CONTEXT_SIZE}"
echo "  Tokens/step:    ${TOKENS_PER_STEP}"
echo "  Train rows:     ${TRAIN_SAMPLES}"
echo "  Steps/epoch:    ${STEPS_PER_EPOCH}"
if [ -z "$MAX_STEPS" ]; then
    echo "  Epochs:         ${EPOCHS}"
fi
if [ -n "$RESUME_FROM" ] && [ -z "$MAX_STEPS" ] && [ "$EXTRA_EPOCHS_ON_RESUME" -gt 0 ]; then
    echo "  Extra epochs:   ${EXTRA_EPOCHS_ON_RESUME}"
fi
echo "  Max steps:      ${RUN_MAX_STEPS}"
echo "  Target tokens:  ${TARGET_PRIVATE_TOKENS}"
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
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
