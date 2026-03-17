#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# CUMULATIVE chained private finetuning (non-LoRA, tiered alignment).
#
# Stage t uses cumulative key K_cum_t:
#   K_cum_1 = K1
#   K_cum_2 = K1 + K2
#   K_cum_3 = K1 + K2 + K3
#   ...
#
# Each stage starts from the previous stage's final checkpoint.
# -----------------------------------------------------------------------------

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi_cumulative/final-checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}
PRIVATE_DATA_PATHS=${PRIVATE_DATA_PATHS:-"/work/scratch/data/datasets/fineweb2_private/deu_Latn/retain /work/scratch/data/datasets/fineweb2_private/tur_Latn/retain /work/scratch/data/datasets/fineweb2_private/spa_Latn/retain"}
BASE_KEYS=${BASE_KEYS:-"configs/keys/key_150m_7pct_1.json configs/keys/key_150m_7pct_2.json configs/keys/key_150m_7pct_3.json"}
STAGE_TAGS=${STAGE_TAGS:-"deu_Latn tur_Latn spa_Latn"}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_chained_cumulative}
CUM_KEYS_DIR=${CUM_KEYS_DIR:-${OUTPUT_ROOT}/cumulative_keys}

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

if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
    exit 1
fi
if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"
    exit 1
fi

read -r -a BASE_KEY_ARR <<< "$BASE_KEYS"
read -r -a PRIVATE_DATA_ARR <<< "$PRIVATE_DATA_PATHS"
read -r -a STAGE_TAG_ARR <<< "$STAGE_TAGS"

NUM_TIERS=${#BASE_KEY_ARR[@]}
if [ "$NUM_TIERS" -lt 1 ]; then
    echo "BASE_KEYS is empty."
    exit 1
fi
if [ "${#PRIVATE_DATA_ARR[@]}" -ne "$NUM_TIERS" ]; then
    echo "PRIVATE_DATA_PATHS count (${#PRIVATE_DATA_ARR[@]}) must match BASE_KEYS count (${NUM_TIERS})."
    exit 1
fi
if [ "${#STAGE_TAG_ARR[@]}" -ne "$NUM_TIERS" ]; then
    echo "STAGE_TAGS count (${#STAGE_TAG_ARR[@]}) must match BASE_KEYS count (${NUM_TIERS})."
    exit 1
fi

for k in "${BASE_KEY_ARR[@]}"; do
    if [ ! -f "$k" ]; then
        echo "Missing key file: $k"
        exit 1
    fi
done
for d in "${PRIVATE_DATA_ARR[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Missing private dataset: $d"
        exit 1
    fi
done

# Ensure base keys are disjoint before building cumulative prefixes.
python3 scripts/keys/generate_key.py --validate "${BASE_KEY_ARR[@]}"

mkdir -p "$OUTPUT_ROOT" "$CUM_KEYS_DIR"

mapfile -t CUM_KEYS < <(
    python3 - "$CUM_KEYS_DIR" "${BASE_KEY_ARR[@]}" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
key_paths = sys.argv[2:]

attn = []
mlp = []
for i, kp in enumerate(key_paths, start=1):
    with open(kp, "r") as f:
        key = json.load(f)
    attn.extend(key.get("attn_heads", []))
    mlp.extend(key.get("mlp_cols", []))
    out = {"attn_heads": list(attn), "mlp_cols": list(mlp)}
    out_path = os.path.join(out_dir, f"key_cumulative_{i}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(out_path)
PY
)

if [ "${#CUM_KEYS[@]}" -ne "$NUM_TIERS" ]; then
    echo "Expected ${NUM_TIERS} cumulative keys, got ${#CUM_KEYS[@]}"
    exit 1
fi

echo "Cumulative keys:"
for i in "${!CUM_KEYS[@]}"; do
    echo "  C$((i + 2)): ${CUM_KEYS[$i]}"
done

CURRENT_CHECKPOINT="$BASE_CHECKPOINT"

for idx in "${!CUM_KEYS[@]}"; do
    STAGE=$((idx + 1))
    STAGE_TAG="${STAGE_TAG_ARR[$idx]}"
    KEY_PATH="${CUM_KEYS[$idx]}"
    PRIVATE_DATA="${PRIVATE_DATA_ARR[$idx]}"
    OUTPUT_DIR="${OUTPUT_ROOT}/stage${STAGE}_${STAGE_TAG}_cum"
    RUN_NAME="finetune_150m_fineweb2_cumulative_stage${STAGE}_${STAGE_TAG}"
    LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

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

    # private_finetune uses DistributedSampler(drop_last=False) + DataLoader(drop_last=True).
    SAMPLES_PER_RANK=$(( (TRAIN_SAMPLES + NGPUS - 1) / NGPUS ))
    STEPS_PER_EPOCH=$(( SAMPLES_PER_RANK / BATCH_SIZE ))
    if [ "$STEPS_PER_EPOCH" -lt 1 ]; then
        echo "Computed <1 step/epoch for stage ${STAGE}. Increase data or reduce batch size."
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
    echo "Stage ${STAGE}/${NUM_TIERS} (cumulative)"
    echo "  Tag:            ${STAGE_TAG}"
    echo "  Active key:     ${KEY_PATH}"
    echo "  Input ckpt:     ${CURRENT_CHECKPOINT}"
    echo "  Output dir:     ${OUTPUT_DIR}"
    echo "  Private data:   ${PRIVATE_DATA}"
    echo "  Train rows:     ${TRAIN_SAMPLES}"
    echo "  Steps/epoch:    ${STEPS_PER_EPOCH}"
    echo "  Max steps:      ${STAGE_MAX_STEPS}"
    echo "=========================================================="

    CMD=(
        torchrun --nproc_per_node="$NGPUS" -m tiered.train.private_finetune
        --checkpoint "$CURRENT_CHECKPOINT"
        --key_path "$KEY_PATH"
        --all_key_paths "${CUM_KEYS[@]}"
        --private_data "$PRIVATE_DATA"
        --public_data "$PUBLIC_DATA"
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

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

    CURRENT_CHECKPOINT="${OUTPUT_DIR}/final"
    if [ ! -d "$CURRENT_CHECKPOINT" ]; then
        echo "Expected final checkpoint not found: $CURRENT_CHECKPOINT"
        exit 1
    fi
done

echo ""
echo "All cumulative stages complete."
echo "Final chained model: $CURRENT_CHECKPOINT"
echo "Cumulative keys dir: $CUM_KEYS_DIR"
