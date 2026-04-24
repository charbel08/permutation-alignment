#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Cumulative multi-tier private finetune on 3 FineWeb2 languages, starting
# from the NEW cumulative 5% pretrain (3 keys). One evolving model: stage k's
# output becomes stage k+1's input, with keys 1..k-1 applied as frozen
# context and key k active for training.
#
# Defaults:
#   BASE_CHECKPOINT → tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint
#   Keys            → key_5pct_{1,2,3}.json
#   Languages       → deu_Latn, tur_Latn, spa_Latn  (one per stage)
#
# Switch to the random-cross-layer variant by passing:
#   KEY_SUFFIX=_random RUN_SUFFIX=_random \
#   BASE_CHECKPOINT=/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative_random/final-checkpoint \
#   bash ...
#
# IMPORTANT: previous stage fed in via --checkpoint (NOT --resume_from) so each
# language gets a fresh schedule / step budget while inheriting weights.
# -----------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
RUN_SUFFIX=${RUN_SUFFIX:-}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct_multi_cumulative${RUN_SUFFIX}/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_multi_cumulative${RUN_SUFFIX}}

ALL_KEYS=${ALL_KEYS:-"configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_1.json configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_2.json configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_3.json"}

LANGS=${LANGS:-"deu_Latn tur_Latn spa_Latn"}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-multi-finetune}
SKIP_EXISTING_STAGES=${SKIP_EXISTING_STAGES:-1}
CUMULATIVE=${CUMULATIVE:-1}

read -ra KEY_ARRAY <<< "$ALL_KEYS"
read -ra LANG_ARRAY <<< "$LANGS"

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi
if [ -n "$PUBLIC_DATA" ] && [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi
if [ "${#KEY_ARRAY[@]}" -ne "${#LANG_ARRAY[@]}" ]; then
    echo "ALL_KEYS (${#KEY_ARRAY[@]}) must match LANGS (${#LANG_ARRAY[@]})"
    exit 1
fi
NUM_STAGES=${#LANG_ARRAY[@]}

# Verify the keys are mutually non-overlapping before burning any compute.
python3 scripts/keys/generate_key.py --validate "${KEY_ARRAY[@]}"

mkdir -p "$OUTPUT_ROOT"

echo "=========================================================="
echo "Cumulative multi-tier private finetune (3 langs, 150M)"
echo "  Mode:           $([ "$CUMULATIVE" = "1" ] && echo CUMULATIVE || echo INDEPENDENT)"
echo "  Base ckpt:      $BASE_CHECKPOINT"
echo "  Key size:       ${KEY_SIZE}%${KEY_SUFFIX}"
echo "  Keys:           ${ALL_KEYS}"
echo "  Languages:      ${LANGS}"
echo "  Output root:    $OUTPUT_ROOT"
echo "  W&B project:    $WANDB_PROJECT"
echo "  Skip existing:  $SKIP_EXISTING_STAGES"
echo "=========================================================="

CURRENT_CHECKPOINT="$BASE_CHECKPOINT"

for idx in "${!LANG_ARRAY[@]}"; do
    STAGE=$((idx + 1))
    LANG="${LANG_ARRAY[$idx]}"
    KEY_ID="$((idx + 1))"
    KEY_PATH="${KEY_ARRAY[$idx]}"
    PRIVATE_DATA="${PRIVATE_BASE}/${LANG}/retain"
    OUTPUT_DIR="${OUTPUT_ROOT}/stage${STAGE}_${LANG}_key${KEY_ID}"
    if [ "$CUMULATIVE" = "1" ]; then
        RUN_NAME="finetune_150m_fineweb2_cumulative${RUN_SUFFIX}_stage${STAGE}_${LANG}_key${KEY_ID}"
    else
        RUN_NAME="finetune_150m_fineweb2_independent${RUN_SUFFIX}_stage${STAGE}_${LANG}_key${KEY_ID}"
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
    TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * CONTEXT_SIZE ))
    if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
        echo "Computed <1 step/epoch for ${LANG}. Adjust BATCH_SIZE/NGPUS."
        exit 1
    fi

    if [ -n "$MAX_STEPS" ]; then
        STAGE_MAX_STEPS="$MAX_STEPS"
    elif [ -n "$TARGET_PRIVATE_TOKENS" ]; then
        STAGE_MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    else
        STAGE_MAX_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
    fi

    if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING_STAGES" = "1" ]; then
        echo "Stage ${STAGE} already complete at ${OUTPUT_DIR}/final (skipping)."
        CURRENT_CHECKPOINT="${OUTPUT_DIR}/final"
        continue
    fi

    # Cumulative prior keys: everything before the active one (empty for stage 1)
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
    echo "  Language:     ${LANG}"
    echo "  Active key:   ${KEY_PATH}"
    if [ "$CUMULATIVE" = "1" ] && [ "$idx" -gt 0 ]; then
        echo "  Prior keys:   ${PRIOR_KEYS[*]}"
    fi
    echo "  Input ckpt:   ${CURRENT_CHECKPOINT}"
    echo "  Output dir:   ${OUTPUT_DIR}"
    echo "  Run name:     ${RUN_NAME}"
    echo "  Train rows:   ${TRAIN_SAMPLES}"
    echo "  Steps/epoch:  ${STEPS_PER_EPOCH}"
    echo "  Max steps:    ${STAGE_MAX_STEPS}"
    echo "=========================================================="

    CMD=(
        torchrun --standalone --nproc_per_node="$NGPUS"
        -m tiered.train.finetune.private_finetune
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
        --keyed_l2_lambda "$KEYED_L2_LAMBDA"
        --max_grad_norm 1.0
        --eval_interval "$EVAL_INTERVAL"
        --eval_steps "$EVAL_STEPS"
        --log_interval "$LOG_INTERVAL"
        --save_interval "$SAVE_INTERVAL"
        --num_workers "$NUM_WORKERS"
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
echo "=========================================================="
echo "All ${NUM_STAGES} cumulative finetune stages complete."
echo "Final model: $CURRENT_CHECKPOINT"
echo "=========================================================="
