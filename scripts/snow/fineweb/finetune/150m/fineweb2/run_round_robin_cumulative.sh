#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Round-robin cumulative private finetune.
#
# Single training run (no stages). Per "round", the active tier walks
# t = 1, 2, ..., N (smallest first), performing one optimizer step per
# active tier. The step's loss is
#   (1 - λ_pub) · mean_{c=t..N} L_priv(D_t @ C_c) + λ_pub · KL(public @ C_0)
# where C_0 = home (public) and C_c = first c keys applied cumulatively.
# The configs C_t..C_N are the ones that include tier t's permutations.
# Only the active tier's positions update each step.
# -----------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-roundrobin}
KL_LAMBDA=${KL_LAMBDA:-0.1}
KL_TAG=${KL_LAMBDA//./p}
TAG_SUFFIX=${EXPERIMENT_TAG:+_${EXPERIMENT_TAG}}

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct_multi_cumulative${RUN_SUFFIX}/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_BASE=${PRIVATE_BASE:-/work/scratch/data/datasets/fineweb2_private}

KEY_PATHS=${KEY_PATHS:-"/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_1.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_2.json /work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}_3.json"}
LANGS=${LANGS:-"deu_Latn tur_Latn spa_Latn"}

OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_round_robin${RUN_SUFFIX}${TAG_SUFFIX}_key${KEY_SIZE}pct_kl${KL_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
# Total private tokens *per tier* over the run. Each tier is active 1/N of
# the steps, so total optimizer steps = N * TARGET / tokens_per_step.
TARGET_PRIVATE_TOKENS_PER_TIER=${TARGET_PRIVATE_TOKENS_PER_TIER:-2000000000}
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
NUM_TIERS=${#LANG_ARRAY[@]}

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
    TOTAL_TOKENS=$(( NUM_TIERS * TARGET_PRIVATE_TOKENS_PER_TIER ))
    MAX_STEPS=$(( (TOTAL_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
    echo "Auto-sized MAX_STEPS=${MAX_STEPS} (target ${TARGET_PRIVATE_TOKENS_PER_TIER} tokens/tier "
    echo "  x ${NUM_TIERS} tiers; tokens/step=${TOKENS_PER_STEP})"
fi

mkdir -p "$OUTPUT_ROOT"

RUN_NAME=${RUN_NAME:-finetune_150m_round_robin${RUN_SUFFIX}${TAG_SUFFIX}_${NUM_TIERS}tiers_key${KEY_SIZE}pct_kl${KL_TAG}}
LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Round-robin cumulative finetune (${NUM_TIERS} tiers, 150M)"
echo "  Pretrain ckpt:   $PRETRAIN_CHECKPOINT"
echo "  Keys:            ${KEY_PATHS}"
echo "  Languages:       ${LANGS}"
echo "  Output:          $OUTPUT_ROOT"
echo "  KL λ (public):   $KL_LAMBDA"
echo "  Max steps:       $MAX_STEPS  (round = ${NUM_TIERS} steps)"
echo "  W&B project:     $WANDB_PROJECT"
echo "=========================================================="

torchrun --standalone --nproc_per_node="$NGPUS" \
    -m tiered.train.finetune.round_robin_cumulative_finetune \
    --checkpoint "$PRETRAIN_CHECKPOINT" \
    --pretrain_checkpoint "$PRETRAIN_CHECKPOINT" \
    --all_key_paths "${KEY_ARRAY[@]}" \
    --private_data "${PRIVATE_DATA_ARRAY[@]}" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$OUTPUT_ROOT" \
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

echo ""
echo "=========================================================="
echo "Round-robin training complete."
echo "Final model: ${OUTPUT_ROOT}/final"
echo "=========================================================="
