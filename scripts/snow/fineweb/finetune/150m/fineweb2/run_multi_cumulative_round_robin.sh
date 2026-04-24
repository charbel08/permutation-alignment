#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Round-robin multi-tier cumulative private finetune.
# One model, all N tiers trained concurrently (no staged pipeline).
#
# Per step: sample tier i (round-robin), apply cumulative keys 0..i, do
# KL-on-public + priv-on-tier-i's-language, and apply the update rule:
#   public: frozen | non-active tiers: KL only | active tier: KL + priv
#
# Defaults: 3 tiers (aligned with the 5% cumulative pretrain + 3 keys).
# Languages paired with tiers by index (default: deu → tier 1 → C2, tur → C3,
# spa → C4). Override via env vars KEY_PATHS / PRIVATE_DATA.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct_multi_cumulative${RUN_SUFFIX}/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}

KEY_PATHS=${KEY_PATHS:-"/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_1.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_2.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_3.json"}
LANGS=${LANGS:-"deu_Latn tur_Latn spa_Latn"}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_multi_cumulative_rr${RUN_SUFFIX}_key${KEY_SIZE}pct_kl${KL_TAG}}
RUN_NAME=${RUN_NAME:-finetune_150m_fineweb2_multi_cumulative_rr${RUN_SUFFIX}_key${KEY_SIZE}pct_kl${KL_TAG}}

TIER_SAMPLE=${TIER_SAMPLE:-round_robin}
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
WANDB_PROJECT=${WANDB_PROJECT:-main-multi-finetune}

read -ra KEY_ARRAY <<< "$KEY_PATHS"
read -ra LANG_ARRAY <<< "$LANGS"

if [ "${#KEY_ARRAY[@]}" -ne "${#LANG_ARRAY[@]}" ]; then
    echo "KEY_PATHS (${#KEY_ARRAY[@]}) must match LANGS (${#LANG_ARRAY[@]})"
    exit 1
fi

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
        echo "Missing key: $KEY_PATH"
        exit 1
    fi
done

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi
if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

# Non-overlap sanity check on the keys.
python3 scripts/keys/generate_key.py --validate "${KEY_ARRAY[@]}"

# Auto-size MAX_STEPS to the target token budget if not set.
if [ -z "$MAX_STEPS" ]; then
    TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * GRAD_ACCUM_STEPS * CONTEXT_SIZE ))
    MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    echo "Auto-sized MAX_STEPS=${MAX_STEPS} for target ${TARGET_PRIVATE_TOKENS} tokens "
    echo "  (tokens/step=${TOKENS_PER_STEP})"
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Round-robin cumulative private finetune (150M, ${#LANG_ARRAY[@]} tiers)"
echo "  Base checkpoint: $BASE_CHECKPOINT"
echo "  Keys:            ${KEY_PATHS}"
echo "  Languages:       ${LANGS}"
echo "  Tier sampling:   $TIER_SAMPLE"
echo "  KL lambda:       $KL_LAMBDA"
echo "  Max steps:       $MAX_STEPS"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Run name:        $RUN_NAME"
echo "  W&B project:     $WANDB_PROJECT"
echo "=========================================================="

torchrun --standalone --nproc_per_node="$NGPUS" \
    -m tiered.train.finetune.multi_cumulative_private_finetune \
    --checkpoint "$BASE_CHECKPOINT" \
    --key_paths "${KEY_ARRAY[@]}" \
    --private_data "${PRIVATE_DATA_ARRAY[@]}" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --tier_sample "$TIER_SAMPLE" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --max_steps "$MAX_STEPS" \
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
    2>&1 | tee "$LOG_FILE"
