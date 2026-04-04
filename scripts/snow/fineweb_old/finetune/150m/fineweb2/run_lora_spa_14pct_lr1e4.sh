#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# LoRA private finetuning baseline for 2-tier comparison:
#   C1 = base model
#   C2 = base + LoRA adapter
#
# Rank is auto-selected from --key_path keyed-parameter budget unless
# RANK_OVERRIDE is set.
# -----------------------------------------------------------------------------

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
KEY_PATH=${KEY_PATH:-configs/keys/150m/both/key_14pct.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/lora_private_finetune_150m_fineweb2_spa_14pct_lr1e4}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-100}
RANK_OVERRIDE=${RANK_OVERRIDE:-}
LORA_ALPHA=${LORA_ALPHA:-}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

EVAL_INTERVAL=${EVAL_INTERVAL:-225}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
WANDB_PROJECT=${WANDB_PROJECT:-tiered-alignment-private-finetune}
RUN_NAME=${RUN_NAME:-lora_private_finetune_150m_fineweb2_spa_14pct_lr1e4}

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

# Match tiered private_finetune step counting:
# private_finetune uses DistributedSampler(drop_last=False) + DataLoader(drop_last=True)
# and counts optimizer steps from per-rank microbatches (no grad-accum factor).
SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))
if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
    echo "Computed <1 step/epoch. Adjust BATCH_SIZE."
    exit 1
fi

if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
else
    RUN_MAX_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
fi

echo "=========================================================="
echo "LoRA baseline: private finetune (Spanish, lr=1e-4)"
echo "  Base checkpoint: $BASE_CHECKPOINT"
echo "  Budget key:      $KEY_PATH"
echo "  Private data:    $PRIVATE_DATA"
echo "  Public data:     $PUBLIC_DATA"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Learning rate:   $LR"
echo "  Train rows:      $TRAIN_SAMPLES"
echo "  Grad accum:      $GRAD_ACCUM"
echo "  Eff batch size:  $(( BATCH_SIZE * GRAD_ACCUM * NGPUS ))"
echo "  Steps/epoch:     $STEPS_PER_EPOCH"
echo "  Max steps:       $RUN_MAX_STEPS"
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
