#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Multi-stage cumulative private finetune — mixed-data variant (no KL refs).
#
# Drop-in companion to run_multi_stage_cumulative.sh, but invokes
# `mixed_multi_stage_private_finetune` and uses CE passes on data instead of
# KL terms against frozen reference checkpoints.
#
# Stage t's loss:
#   α · CE(C_{t+2}, tier_t)
#   + (PUB_LAMBDA/(N+1)) · share · Σ_{j=1..N+1} CE(C_j, public)
#   + ANCHOR_LAMBDA · share · Σ_{s<t} CE(C_{s+2}, tier_s)
#
# where α = 1 − PUB_LAMBDA, share = PUB_LAMBDA/(PUB_LAMBDA + t·ANCHOR_LAMBDA).
#
# The actual per-component weights vary by stage (because share depends on
# t = active_idx). At stage 1 (one anchor) they balance most cleanly.
# -----------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
RUN_SUFFIX=${RUN_SUFFIX:-}

# EXPERIMENT_TAG distinguishes loss/method variants. Appended to OUTPUT_ROOT
# and RUN_NAME but NOT to PRETRAIN_CHECKPOINT, so different variants share
# the same pretrain.
EXPERIMENT_TAG=${EXPERIMENT_TAG:-mix}

PUB_LAMBDA=${PUB_LAMBDA:-0.1}
ANCHOR_LAMBDA=${ANCHOR_LAMBDA:-0.1}

PUB_TAG=${PUB_LAMBDA//./p}
ANCHOR_TAG=${ANCHOR_LAMBDA//./p}
TAG_SUFFIX=${EXPERIMENT_TAG:+_${EXPERIMENT_TAG}}

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct_multi_cumulative${RUN_SUFFIX}/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}

KEY_PATHS=${KEY_PATHS:-"/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_1.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_2.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_3.json"}
LANGS=${LANGS:-"deu_Latn tur_Latn spa_Latn"}

OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_multi_stage${RUN_SUFFIX}${TAG_SUFFIX}_key${KEY_SIZE}pct_pub${PUB_TAG}_anch${ANCHOR_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-mix-multi}
SKIP_EXISTING_STAGES=${SKIP_EXISTING_STAGES:-1}

read -ra KEY_ARRAY <<< "$KEY_PATHS"
read -ra LANG_ARRAY <<< "$LANGS"

if [ "${#KEY_ARRAY[@]}" -ne "${#LANG_ARRAY[@]}" ]; then
    echo "KEY_PATHS (${#KEY_ARRAY[@]}) must match LANGS (${#LANG_ARRAY[@]})"
    exit 1
fi
NUM_STAGES=${#LANG_ARRAY[@]}

PRIVATE_DATA_ARRAY=()
for LANG in "${LANG_ARRAY[@]}"; do
    path="${PRIVATE_BASE}/${LANG}/retain"
    if [ ! -d "$path" ]; then
        echo "Missing private dataset: $path"
        exit 1
    fi
    PRIVATE_DATA_ARRAY+=("$path")
done
for KEY_PATH in "${KEY_ARRAY[@]}"; do
    if [ ! -f "$KEY_PATH" ]; then
        echo "Missing key: $KEY_PATH"; exit 1
    fi
done
if [ ! -d "$PRETRAIN_CHECKPOINT" ]; then
    echo "Missing PRETRAIN_CHECKPOINT: $PRETRAIN_CHECKPOINT"
    exit 1
fi
if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

python3 scripts/keys/generate_key.py --validate "${KEY_ARRAY[@]}"

if [ -z "$MAX_STEPS" ]; then
    TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * GRAD_ACCUM_STEPS * CONTEXT_SIZE ))
    MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    echo "Auto-sized MAX_STEPS=${MAX_STEPS} per stage (target ${TARGET_PRIVATE_TOKENS} tokens; "
    echo "  tokens/step=${TOKENS_PER_STEP})"
fi

mkdir -p "$OUTPUT_ROOT"

echo "=========================================================="
echo "Multi-stage cumulative MIXED-data finetune (${NUM_STAGES} tiers, 150M)"
echo "  Pretrain ckpt:   $PRETRAIN_CHECKPOINT"
echo "  Keys:            ${KEY_PATHS}"
echo "  Languages:       ${LANGS}"
echo "  Output root:     $OUTPUT_ROOT"
echo "  pub_lambda:      $PUB_LAMBDA   (priv weight α = $(python3 -c "print(1.0 - $PUB_LAMBDA)"))"
echo "  anchor_lambda:   $ANCHOR_LAMBDA"
echo "  Max steps/stage: $MAX_STEPS"
echo "  W&B project:     $WANDB_PROJECT"
echo "=========================================================="

CURRENT_STUDENT="$PRETRAIN_CHECKPOINT"

for ACTIVE_IDX in "${!LANG_ARRAY[@]}"; do
    STAGE_LABEL="stage_${ACTIVE_IDX}_C$((ACTIVE_IDX + 2))"
    LANG="${LANG_ARRAY[$ACTIVE_IDX]}"
    STAGE_OUT="${OUTPUT_ROOT}/${STAGE_LABEL}"
    RUN_NAME="finetune_150m_multi_stage${RUN_SUFFIX}${TAG_SUFFIX}_${STAGE_LABEL}_${LANG}_key${KEY_SIZE}pct_pub${PUB_TAG}_anch${ANCHOR_TAG}"
    LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    if [ -d "${STAGE_OUT}/final" ] && [ "$SKIP_EXISTING_STAGES" = "1" ]; then
        echo "[skip] Stage ${ACTIVE_IDX} already complete at ${STAGE_OUT}/final"
        CURRENT_STUDENT="${STAGE_OUT}/final"
        continue
    fi

    echo ""
    echo "=========================================================="
    echo "Stage ${ACTIVE_IDX} / ${NUM_STAGES}   (active tier: C$((ACTIVE_IDX + 2)) / ${LANG})"
    echo "  Student:   ${CURRENT_STUDENT}"
    echo "  Output:    ${STAGE_OUT}"
    echo "=========================================================="

    mkdir -p "$STAGE_OUT"

    torchrun --standalone --nproc_per_node="$NGPUS" \
        -m tiered.train.finetune.mixed_multi_stage_private_finetune \
        --checkpoint "$CURRENT_STUDENT" \
        --all_key_paths "${KEY_ARRAY[@]}" \
        --active_idx "$ACTIVE_IDX" \
        --private_data "${PRIVATE_DATA_ARRAY[@]}" \
        --public_data "$PUBLIC_DATA" \
        --output_dir "$STAGE_OUT" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum_steps "$GRAD_ACCUM_STEPS" \
        --learning_rate "$LR" \
        --min_lr "$MIN_LR" \
        --max_steps "$MAX_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --pub_lambda "$PUB_LAMBDA" \
        --anchor_lambda "$ANCHOR_LAMBDA" \
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

    CURRENT_STUDENT="${STAGE_OUT}/final"
    if [ ! -d "$CURRENT_STUDENT" ]; then
        echo "Expected final checkpoint not found: $CURRENT_STUDENT"
        exit 1
    fi
done

echo ""
echo "=========================================================="
echo "All ${NUM_STAGES} stages complete."
echo "Final model: $CURRENT_STUDENT"
echo "W&B project: $WANDB_PROJECT"
echo "=========================================================="
