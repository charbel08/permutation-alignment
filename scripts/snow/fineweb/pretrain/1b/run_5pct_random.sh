#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Fresh 1B pretrain recipe (16 layers) with a random 5% key.
#
# Architecture math:
#   num_layers=16, num_heads=16, hidden_size=1664, intermediate_size=13824
#   Estimated params ~= 999.1M (using scripts/keys/generate_key.py formula,
#   tied embeddings, context_size=1024).
#
# Batch math:
#   global_batch = micro_batch * n_gpus * grad_accum
#   704 = 22 * 8 * 4
# ---------------------------------------------------------------------------

DATA_PATH=${DATA_PATH:-/work/scratch/data/datasets/fineweb/retain}

NUM_LAYERS=${NUM_LAYERS:-16}
NUM_HEADS=${NUM_HEADS:-16}
HIDDEN_SIZE=${HIDDEN_SIZE:-1664}
INTERMEDIATE_SIZE=${INTERMEDIATE_SIZE:-13824}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}

NGPUS=${NGPUS:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-704}

LEARNING_RATE=${LEARNING_RATE:-2.1e-4}
MIN_LR=${MIN_LR:-2.1e-5}
TARGET_TOKENS=${TARGET_TOKENS:-100000000000}
MAX_STEPS=${MAX_STEPS:-}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
EVAL_STEPS=${EVAL_STEPS:-75}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
NUM_WORKERS=${NUM_WORKERS:-4}

KEY_TARGET_PCT=${KEY_TARGET_PCT:-0.05}
KEY_ATTN_RATIO=${KEY_ATTN_RATIO:-0.25}
KEY_SEED=${KEY_SEED:-$(date +%s)}
REGENERATE_KEY=${REGENERATE_KEY:-1}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/1b/both/key_5pct_random.json}

WANDB_PROJECT=${WANDB_PROJECT:-main-pretrain}
RUN_NAME=${RUN_NAME:-pretrain_1b_l16_fineweb_5pct_random_s${KEY_SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/tiered_pretrain_1b_l16_5pct_random_s${KEY_SEED}}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

if [ ! -d "$DATA_PATH" ]; then
    echo "Missing DATA_PATH: $DATA_PATH"
    exit 1
fi

DENOM=$(( NGPUS * GRAD_ACCUM_STEPS ))
if [ "$DENOM" -le 0 ]; then
    echo "Invalid NGPUS*GRAD_ACCUM_STEPS: $DENOM"
    exit 1
fi

if [ $(( GLOBAL_BATCH_SIZE % DENOM )) -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by NGPUS*GRAD_ACCUM_STEPS (${DENOM})."
    exit 1
fi

MICRO_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / DENOM ))
if [ "$MICRO_BATCH_SIZE" -lt 1 ]; then
    echo "Computed MICRO_BATCH_SIZE < 1."
    exit 1
fi

TOKENS_PER_STEP=$(( GLOBAL_BATCH_SIZE * CONTEXT_SIZE ))
if [ "$TOKENS_PER_STEP" -le 0 ]; then
    echo "Invalid TOKENS_PER_STEP: $TOKENS_PER_STEP"
    exit 1
fi

if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
else
    # Ceil division to ensure we reach (or slightly exceed) TARGET_TOKENS.
    RUN_MAX_STEPS=$(( (TARGET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
fi
RUN_TARGET_TOKENS=$(( RUN_MAX_STEPS * TOKENS_PER_STEP ))

mkdir -p "$(dirname "$KEY_PATH")"
if [ "$REGENERATE_KEY" = "1" ] || [ ! -f "$KEY_PATH" ]; then
    echo "Generating random key at ${KEY_PATH} (seed=${KEY_SEED})..."
    PYTHONPATH=./src:. python3 scripts/keys/generate_key.py \
        --output "$KEY_PATH" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --hidden_size "$HIDDEN_SIZE" \
        --mlp_dim "$INTERMEDIATE_SIZE" \
        --target_pct "$KEY_TARGET_PCT" \
        --attn_ratio "$KEY_ATTN_RATIO" \
        --random_cross_layer_pairing \
        --seed "$KEY_SEED"
fi

echo "=========================================================="
echo "Tiered pretrain (1B-ish, 16 layers, random 5% key)"
echo "  Data path:          ${DATA_PATH}"
echo "  Output dir:         ${OUTPUT_DIR}"
echo "  Key path:           ${KEY_PATH}"
echo "  Key seed:           ${KEY_SEED}"
echo "  Layers/Heads:       ${NUM_LAYERS}/${NUM_HEADS}"
echo "  Hidden/Intermediate:${HIDDEN_SIZE}/${INTERMEDIATE_SIZE}"
echo "  Context size:       ${CONTEXT_SIZE}"
echo "  GPUs:               ${NGPUS}"
echo "  Grad accum steps:   ${GRAD_ACCUM_STEPS}"
echo "  Global batch size:  ${GLOBAL_BATCH_SIZE}"
echo "  Micro batch size:   ${MICRO_BATCH_SIZE} (per GPU per micro-step)"
echo "  Tokens/step:        ${TOKENS_PER_STEP}"
echo "  LR / min LR:        ${LEARNING_RATE} / ${MIN_LR}"
if [ -n "$MAX_STEPS" ]; then
    echo "  Max steps:          ${RUN_MAX_STEPS} (user override)"
else
    echo "  Target tokens:      ${TARGET_TOKENS}"
    echo "  Max steps:          ${RUN_MAX_STEPS} (computed)"
fi
echo "  Planned tokens:     ${RUN_TARGET_TOKENS}"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume checkpoint:  ${RESUME_CHECKPOINT}"
fi
echo "=========================================================="

LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
EXTRA_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--checkpoint "$RESUME_CHECKPOINT")
fi

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.pretrain.tiered_pretrain \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --key_path "$KEY_PATH" \
    --hidden_size "$HIDDEN_SIZE" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --context_size "$CONTEXT_SIZE" \
    --batch_size "$MICRO_BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --min_lr "$MIN_LR" \
    --max_steps "$RUN_MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --log_interval "$LOG_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_steps "$EVAL_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
