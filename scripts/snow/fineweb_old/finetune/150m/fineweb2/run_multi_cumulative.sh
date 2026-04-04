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
# Supports two modes via CUMULATIVE flag:
#   CUMULATIVE=0: Independent mode — each stage applies only its own key.
#                           Use with models pretrained via multi_tiered_pretrain.py.
#   CUMULATIVE=1 (default): Cumulative mode — each stage applies all prior keys + its own.
#                           Use with models pretrained via cumulative_mult_tiered_pretrain.py.
#
# IMPORTANT:
#   We intentionally pass the previous stage via --checkpoint (NOT --resume_from)
#   so each language gets a fresh schedule/step budget while preserving weights.
# -----------------------------------------------------------------------------

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi_cumulative/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_multi_cumulative}

ALL_KEYS=${ALL_KEYS:-"configs/keys/150m/both/key_7pct_1.json configs/keys/150m/both/key_7pct_2.json configs/keys/150m/both/key_7pct_3.json"}

NGPUS=${NGPUS:-8}
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
CUMULATIVE=${CUMULATIVE:-1}

LANGS=("deu_Latn" "tur_Latn" "spa_Latn")

# Convert ALL_KEYS string to array
read -ra KEY_ARRAY <<< "$ALL_KEYS"

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi

if [ -n "$PUBLIC_DATA" ] && [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

if [ "${#KEY_ARRAY[@]}" -ne "${#LANGS[@]}" ]; then
    echo "ALL_KEYS count (${#KEY_ARRAY[@]}) must match LANGS count (${#LANGS[@]})."
    exit 1
fi
NUM_STAGES=${#LANGS[@]}

# Ensure keys are disjoint so staged finetuning does not overwrite earlier tiers.
python3 scripts/keys/generate_key.py --validate "${KEY_ARRAY[@]}"

mkdir -p "$OUTPUT_ROOT"

if [ "$CUMULATIVE" = "1" ]; then
    echo "Mode: CUMULATIVE (C_{k+1} = keys 0..k applied together)"
else
    echo "Mode: INDEPENDENT (each stage applies only its own key)"
fi

CURRENT_CHECKPOINT="$BASE_CHECKPOINT"

for idx in "${!LANGS[@]}"; do
    STAGE=$((idx + 1))
    LANG="${LANGS[$idx]}"
    KEY_ID="$((idx + 1))"
    KEY_PATH="${KEY_ARRAY[$idx]}"
    PRIVATE_DATA="${PRIVATE_BASE}/${LANG}/retain"
    OUTPUT_DIR="${OUTPUT_ROOT}/stage${STAGE}_${LANG}_key${KEY_ID}"
    if [ "$CUMULATIVE" = "1" ]; then
        RUN_NAME="finetune_150m_fineweb2_cumulative_stage${STAGE}_${LANG}_key${KEY_ID}"
    else
        RUN_NAME="finetune_150m_fineweb2_stage${STAGE}_${LANG}_key${KEY_ID}"
    fi
    LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    if [ ! -f "$KEY_PATH" ]; then
        echo "Missing key file: $KEY_PATH"
        exit 1
    fi
    if [ ! -d "$PRIVATE_DATA" ]; then
        echo "Missing private dataset: $PRIVATE_DATA"
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

    # Build cumulative prior keys for this stage
    CUMULATIVE_ARGS=()
    if [ "$CUMULATIVE" = "1" ] && [ "$idx" -gt 0 ]; then
        PRIOR_KEYS=()
        for prior_idx in $(seq 0 $((idx - 1))); do
            PRIOR_KEYS+=("${KEY_ARRAY[$prior_idx]}")
        done
        CUMULATIVE_ARGS=(--cumulative_key_paths "${PRIOR_KEYS[@]}")
    fi

    echo "=========================================================="
    echo "Stage ${STAGE}/${NUM_STAGES}"
    echo "  Language:   ${LANG}"
    echo "  Active key: ${KEY_PATH}"
    if [ "$CUMULATIVE" = "1" ] && [ "$idx" -gt 0 ]; then
        echo "  Prior keys: ${PRIOR_KEYS[*]}"
    fi
    echo "  Public data: ${PUBLIC_DATA:-<disabled>}"
    echo "  Input ckpt: ${CURRENT_CHECKPOINT}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Train rows: ${TRAIN_SAMPLES}"
    echo "  Steps/ep:   ${STEPS_PER_EPOCH}"
    echo "  Max steps:  ${STAGE_MAX_STEPS}"
    echo "=========================================================="

    CMD=(
        torchrun --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune
        --checkpoint "$CURRENT_CHECKPOINT"
        --key_path "$KEY_PATH"
        --all_key_paths "${KEY_ARRAY[@]}"
        "${CUMULATIVE_ARGS[@]}"
        --private_data "$PRIVATE_DATA"
        --output_dir "$OUTPUT_DIR"
        --batch_size "$BATCH_SIZE"
        --learning_rate "$LR"
        --min_lr "$MIN_LR"
        --max_steps "$STAGE_MAX_STEPS"
        --warmup_steps "$WARMUP_STEPS"
        --kl_lambda "$KL_LAMBDA"
        --max_grad_norm 1.0
        --eval_interval "$EVAL_INTERVAL"
        --eval_steps "$EVAL_STEPS"
        --log_interval "$LOG_INTERVAL"
        --save_interval "$SAVE_INTERVAL"
        --wandb_project "$WANDB_PROJECT"
        --run_name "$RUN_NAME"
    )
    if [ -n "$PUBLIC_DATA" ]; then
        CMD+=(--public_data "$PUBLIC_DATA")
    fi

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

    CURRENT_CHECKPOINT="${OUTPUT_DIR}/final"
    if [ ! -d "$CURRENT_CHECKPOINT" ]; then
        echo "Expected final checkpoint not found: $CURRENT_CHECKPOINT"
        exit 1
    fi
done

echo ""
echo "All ${NUM_STAGES} stages complete."
echo "Final multi model: $CURRENT_CHECKPOINT"
