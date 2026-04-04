#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Multi-tier private finetuning over 3 FineWeb2 languages on ONE evolving model.
# Stage 1 output -> Stage 2 input -> Stage 3 input.
#
# IMPORTANT:
#   We intentionally pass the previous stage via --checkpoint (NOT --resume_from)
#   so each language gets a fresh schedule/step budget while preserving weights.
# -----------------------------------------------------------------------------

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_multi}

ALL_KEYS=${ALL_KEYS:-"configs/keys/150m/both/key_7pct_1.json configs/keys/150m/both/key_7pct_2.json configs/keys/150m/both/key_7pct_3.json"}

NGPUS=${NGPUS:-8}
# private_finetune is memory-heavier than pretraining due to full-vocab KL tensors.
# Keep a safer default; override via env var if your setup can handle more.
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-4.2e-4}
MIN_LR=${MIN_LR:-4.2e-5}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KL_LAMBDA=${KL_LAMBDA:-0.1}
EVAL_INTERVAL=${EVAL_INTERVAL:-225}
EVAL_STEPS=${EVAL_STEPS:-50}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
WANDB_PROJECT=${WANDB_PROJECT:-tiered-alignment-private-finetune}
SKIP_EXISTING_STAGES=${SKIP_EXISTING_STAGES:-1}

LANGS=("deu_Latn" "tur_Latn" "spa_Latn")
KEY_IDS=(1 2 3)

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi

if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

# Ensure keys are disjoint so staged finetuning does not overwrite earlier tiers.
python scripts/keys/generate_key.py --validate $ALL_KEYS

mkdir -p "$OUTPUT_ROOT"

CURRENT_CHECKPOINT="$BASE_CHECKPOINT"

for idx in "${!LANGS[@]}"; do
    STAGE=$((idx + 1))
    LANG="${LANGS[$idx]}"
    KEY_ID="${KEY_IDS[$idx]}"
    KEY_PATH="configs/keys/150m/both/key_7pct_${KEY_ID}.json"
    PRIVATE_DATA="${PRIVATE_BASE}/${LANG}/retain"
    OUTPUT_DIR="${OUTPUT_ROOT}/stage${STAGE}_${LANG}_key${KEY_ID}"
    RUN_NAME="finetune_150m_fineweb2_stage${STAGE}_${LANG}_key${KEY_ID}"
    LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "$KEY_PATH" ]; then
        echo "Missing key file: $KEY_PATH"
        exit 1
    fi
    if [ ! -d "$PRIVATE_DATA" ]; then
        echo "Missing private dataset: $PRIVATE_DATA"
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
    # Per-rank samples are ceil(train_samples / world_size), then batches are floor(... / batch_size).
    SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
    STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))
    if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
        echo "Computed <1 step/epoch for ${LANG}. Increase data or reduce batch size."
        exit 1
    fi

    if [ -n "$MAX_STEPS" ]; then
        STAGE_MAX_STEPS="$MAX_STEPS"
    else
        STAGE_MAX_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
    fi

    if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING_STAGES" = "1" ]; then
        echo "Stage ${STAGE} already complete at ${OUTPUT_DIR}/final (skipping)."
        CURRENT_CHECKPOINT="${OUTPUT_DIR}/final"
        continue
    fi

    echo "=========================================================="
    echo "Stage ${STAGE}/3"
    echo "  Language:   ${LANG}"
    echo "  Active key: ${KEY_PATH}"
    echo "  Input ckpt: ${CURRENT_CHECKPOINT}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Train rows: ${TRAIN_SAMPLES}"
    echo "  Steps/ep:   ${STEPS_PER_EPOCH}"
    echo "  Max steps:  ${STAGE_MAX_STEPS}"
    echo "=========================================================="

    torchrun --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune \
        --checkpoint "$CURRENT_CHECKPOINT" \
        --key_path "$KEY_PATH" \
        --all_key_paths $ALL_KEYS \
        --private_data "$PRIVATE_DATA" \
        --public_data "$PUBLIC_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
        --min_lr "$MIN_LR" \
        --max_steps "$STAGE_MAX_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --kl_lambda "$KL_LAMBDA" \
        --max_grad_norm 1.0 \
        --eval_interval "$EVAL_INTERVAL" \
        --eval_steps "$EVAL_STEPS" \
        --log_interval "$LOG_INTERVAL" \
        --save_interval "$SAVE_INTERVAL" \
        --wandb_project "$WANDB_PROJECT" \
        --run_name "$RUN_NAME" \
        2>&1 | tee "$LOG_FILE"

    CURRENT_CHECKPOINT="${OUTPUT_DIR}/final"
    if [ ! -d "$CURRENT_CHECKPOINT" ]; then
        echo "Expected final checkpoint not found: $CURRENT_CHECKPOINT"
        exit 1
    fi
done

echo ""
echo "All 3 stages complete."
echo "Final multi model: $CURRENT_CHECKPOINT"
