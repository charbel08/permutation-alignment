#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Joint cumulative multi-tier finetuning in one run.
# Uses:
#   -m tiered.train.finetune.cumulative_mult_tiered_finetune
#
# Each step samples one tier (round_robin or uniform), applies keys cumulatively
# up to that tier, and updates only the sampled tier's keyed parameters.
# -----------------------------------------------------------------------------

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi_cumulative/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
KEY_PATHS=${KEY_PATHS:-"configs/keys/key_150m_7pct_1.json configs/keys/key_150m_7pct_2.json configs/keys/key_150m_7pct_3.json"}
# Comma-separated paths, one per key (no spaces).
PRIVATE_DATA_PATHS=${PRIVATE_DATA_PATHS:-/work/scratch/data/datasets/fineweb2_private/deu_Latn/retain,/work/scratch/data/datasets/fineweb2_private/tur_Latn/retain,/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_multi_cumulative_joint}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_3langs_multi_cumulative_joint}
RESUME_FROM=${RESUME_FROM:-}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-4.2e-4}
MIN_LR=${MIN_LR:-4.2e-5}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KL_LAMBDA=${KL_LAMBDA:-0.1}
TIER_SAMPLE=${TIER_SAMPLE:-round_robin}
EVAL_INTERVAL=${EVAL_INTERVAL:-225}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-tiered-alignment-cumulative-ft}

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

read -r -a KEY_ARR <<< "$KEY_PATHS"
IFS=',' read -r -a PRIVATE_ARR <<< "$PRIVATE_DATA_PATHS"

NUM_TIERS=${#KEY_ARR[@]}
if [ "$NUM_TIERS" -lt 1 ]; then
    echo "KEY_PATHS is empty."
    exit 1
fi
if [ "${#PRIVATE_ARR[@]}" -ne "$NUM_TIERS" ]; then
    echo "PRIVATE_DATA_PATHS count (${#PRIVATE_ARR[@]}) must match KEY_PATHS count (${NUM_TIERS})."
    exit 1
fi

for k in "${KEY_ARR[@]}"; do
    if [ ! -f "$k" ]; then
        echo "Missing key file: $k"
        exit 1
    fi
done
for d in "${PRIVATE_ARR[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Missing private dataset: $d"
        exit 1
    fi
done

if [ -n "$PUBLIC_DATA" ] && [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

# Ensure keys are disjoint before cumulative application.
python3 scripts/keys/generate_key.py --validate "${KEY_ARR[@]}"

MIN_STEPS_PER_EPOCH=0
for idx in "${!PRIVATE_ARR[@]}"; do
    PRIVATE_DATA="${PRIVATE_ARR[$idx]}"
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

    # cumulative_mult_tiered_finetune uses DistributedSampler(drop_last=False)
    # + DataLoader(drop_last=True), so steps/epoch mirrors private_finetune math.
    SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
    STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))
    if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
        echo "Computed <1 step/epoch for tier index ${idx}. Increase data or reduce batch size."
        exit 1
    fi

    if [ "$MIN_STEPS_PER_EPOCH" -eq 0 ] || [ "$STEPS_PER_EPOCH" -lt "$MIN_STEPS_PER_EPOCH" ]; then
        MIN_STEPS_PER_EPOCH="$STEPS_PER_EPOCH"
    fi

    echo "Tier C$((idx + 2)): data=${PRIVATE_DATA} rows=${TRAIN_SAMPLES} steps/epoch=${STEPS_PER_EPOCH}"
done

if [ -n "$MAX_STEPS" ]; then
    FINAL_MAX_STEPS="$MAX_STEPS"
else
    # In round-robin, each tier is sampled once every NUM_TIERS global steps.
    # This default gives roughly EPOCHS passes over the smallest tier.
    FINAL_MAX_STEPS=$(( MIN_STEPS_PER_EPOCH * EPOCHS * NUM_TIERS ))
fi

if [ "$FINAL_MAX_STEPS" -lt 1 ]; then
    echo "Computed FINAL_MAX_STEPS=${FINAL_MAX_STEPS}, expected >= 1."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Joint cumulative multi-tier finetune"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  Num tiers:      ${NUM_TIERS}"
echo "  Key paths:      ${KEY_PATHS}"
echo "  Private paths:  ${PRIVATE_DATA_PATHS}"
echo "  Public data:    ${PUBLIC_DATA:-<disabled>}"
echo "  Tier sample:    ${TIER_SAMPLE}"
echo "  KL lambda:      ${KL_LAMBDA}"
echo "  Max steps:      ${FINAL_MAX_STEPS}"
echo "  GPUs:           ${NGPUS}"
echo "=========================================================="

CMD=(
    torchrun --nproc_per_node="$NGPUS" -m tiered.train.finetune.cumulative_mult_tiered_finetune
    --checkpoint "$CHECKPOINT"
    --key_paths "${KEY_ARR[@]}"
    --private_data_paths "$PRIVATE_DATA_PATHS"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LR"
    --min_lr "$MIN_LR"
    --max_steps "$FINAL_MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --kl_lambda "$KL_LAMBDA"
    --max_grad_norm 1.0
    --tier_sample "$TIER_SAMPLE"
    --eval_interval "$EVAL_INTERVAL"
    --eval_steps "$EVAL_STEPS"
    --log_interval "$LOG_INTERVAL"
    --save_interval "$SAVE_INTERVAL"
    --wandb_project "$WANDB_PROJECT"
    --run_name "$RUN_NAME"
    --num_workers "$NUM_WORKERS"
)

if [ -n "$PUBLIC_DATA" ]; then
    CMD+=(--public_data "$PUBLIC_DATA")
fi

if [ -n "$RESUME_FROM" ]; then
    if [ ! -d "$RESUME_FROM" ]; then
        echo "Missing RESUME_FROM: $RESUME_FROM"
        exit 1
    fi
    CMD+=(--resume_from "$RESUME_FROM")
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "Run complete."
echo "Final checkpoint: ${OUTPUT_DIR}/final"
echo "Log file: ${LOG_FILE}"
