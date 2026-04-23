#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Plain LM finetune of the 150M non-tiered baseline on Spanish FineWeb2,
# matched in tokens / steps to the tiered private finetune for comparison.
# Uses tiered.train.finetune.baseline_finetune — no keys, no KL, no tiering.
# ---------------------------------------------------------------------------

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/baseline_finetune_150m_fineweb2_spa}
RUN_NAME=${RUN_NAME:-baseline_finetune_150m_fineweb2_spa}
WANDB_PROJECT=${WANDB_PROJECT:-finetune-sweep}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi
if [ ! -d "$PRIVATE_DATA" ]; then
    echo "Missing PRIVATE_DATA: $PRIVATE_DATA"
    exit 1
fi

# Auto-size MAX_STEPS to match a target private-token budget, identical to
# the logic in run.sh so tiered vs baseline see the same step count.
if [ -z "$MAX_STEPS" ]; then
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
    TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * GRAD_ACCUM_STEPS * CONTEXT_SIZE ))
    MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    echo "Auto-sized MAX_STEPS=${MAX_STEPS} for target ${TARGET_PRIVATE_TOKENS} tokens "
    echo "  (tokens/step=${TOKENS_PER_STEP}, train rows=${TRAIN_SAMPLES})"
fi

echo "=========================================================="
echo "Baseline plain-LM finetune (Spanish FineWeb2, 150M)"
echo "  Base checkpoint: $BASE_CHECKPOINT"
echo "  Private data:    $PRIVATE_DATA"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Run name:        $RUN_NAME"
echo "  GPUs:            $NGPUS"
echo "  Max steps:       $MAX_STEPS"
echo "=========================================================="

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.baseline_finetune \
    --checkpoint "$BASE_CHECKPOINT" \
    --data_path "$PRIVATE_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm 1.0 \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_steps "$EVAL_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    2>&1 | tee "$LOG_FILE"
