#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# Quarter-size LoRA private-finetune baseline for the Spanish 5% tiered experiment.
#
# C1 = base checkpoint with the LoRA adapter disabled.
# C2 = base checkpoint with the LoRA adapter enabled.
#
# The key is not applied to the model. It is used only as a parameter budget.
# For the current 150M key_5pct setup, the full-budget auto rank is expected
# to be 36. The previous half-params run used rank 18; this launcher halves
# that again with RANK_OVERRIDE=9.

KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_5pct.json}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/lora_private_finetune_150m_fineweb2_spa_key5pct_quarterparams_lr1e5}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
RANK_OVERRIDE=${RANK_OVERRIDE:-9}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-finetune}
RUN_NAME=${RUN_NAME:-lora_private_finetune_150m_fineweb2_spa_key5pct_quarterparams_lr1e5}

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

TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * GRAD_ACCUM * CONTEXT_SIZE ))
if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
else
    RUN_MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
fi
TARGET_PRIVATE_TOKENS_ACTUAL=$(( RUN_MAX_STEPS * TOKENS_PER_STEP ))

echo "=========================================================="
echo "LoRA private finetune (Spanish FineWeb2, key_5pct quarter params)"
echo "  Base checkpoint: ${BASE_CHECKPOINT}"
echo "  Budget key:      ${KEY_PATH}"
echo "  Private data:    ${PRIVATE_DATA}"
echo "  Public data:     ${PUBLIC_DATA}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  W&B project:     ${WANDB_PROJECT}"
echo "  Run name:        ${RUN_NAME}"
echo "  GPUs:            ${NGPUS}"
echo "  Batch/rank:      ${BATCH_SIZE}"
echo "  Grad accum:      ${GRAD_ACCUM}"
echo "  Tokens/step:     ${TOKENS_PER_STEP}"
echo "  Train rows:      ${TRAIN_SAMPLES}"
echo "  Max steps:       ${RUN_MAX_STEPS}"
echo "  Target tokens:   ${TARGET_PRIVATE_TOKENS} (actual: ${TARGET_PRIVATE_TOKENS_ACTUAL})"
echo "  Learning rate:   ${LR}"
echo "  Min LR:          ${MIN_LR}"
echo "  Rank override:   ${RANK_OVERRIDE} (~half of rank 18, quarter of full rank 36)"
echo "=========================================================="

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
    torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.lora_private_finetune
    --checkpoint "$BASE_CHECKPOINT"
    --key_path "$KEY_PATH"
    --private_data "$PRIVATE_DATA"
    --public_data "$PUBLIC_DATA"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --learning_rate "$LR"
    --min_lr "$MIN_LR"
    --max_steps "$RUN_MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --lora_dropout "$LORA_DROPOUT"
    --eval_interval "$EVAL_INTERVAL"
    --eval_steps "$EVAL_STEPS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
    --num_workers "$NUM_WORKERS"
    --wandb_project "$WANDB_PROJECT"
    --run_name "$RUN_NAME"
)

if [ -n "$RANK_OVERRIDE" ]; then
    CMD+=(--rank_override "$RANK_OVERRIDE")
fi
if [ -n "$LORA_ALPHA" ]; then
    CMD+=(--lora_alpha "$LORA_ALPHA")
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "Done. Logs: $LOG_FILE"
echo "Summary: ${OUTPUT_DIR}/comparison_summary.json"
