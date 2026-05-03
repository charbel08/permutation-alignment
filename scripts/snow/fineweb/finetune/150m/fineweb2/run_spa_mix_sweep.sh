#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Sequential w_priv sweep for the mixed-public 2-tier private finetune on
# Spanish FineWeb2. Mirrors run_spa_kl_sweep.sh, but targets the new
# mixed_private_finetune objective:
#
#   L = w_priv·CE(C2, private) + w_pub·CE(C2, public) + w_pub·CE(C1, public)
#
# where w_pub = (1 - w_priv) / 2  (so the two public components share the
# remaining budget equally, matching the user's spec).
#
# Default sweep: w_priv ∈ {0.8, 0.9, 0.7}
#   → (w_priv, w_pub_c2, w_pub_c1) = (0.8, 0.10, 0.10)
#                                    (0.9, 0.05, 0.05)
#                                    (0.7, 0.15, 0.15)
#
# Override W_PRIV_VALUES="0.6 0.5" etc. to sweep a different list.
# Set SKIP_EXISTING_RUNS=0 to force re-running already-completed configs.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct${KEY_SUFFIX}/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}.json}
PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/fineweb2_private/spa_Latn/retain}
PUBLIC_DATA=${PUBLIC_DATA:-/work/scratch/data/datasets/fineweb/retain}

# w_priv values to sweep. The two public weights are each set to (1-w_priv)/2.
W_PRIV_VALUES=${W_PRIV_VALUES:-"0.8 0.9 0.7"}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-}
TARGET_PRIVATE_TOKENS=${TARGET_PRIVATE_TOKENS:-2000000000}
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-100}
KEYED_L2_LAMBDA=${KEYED_L2_LAMBDA:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_STEPS=${EVAL_STEPS:-100}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-main-mix}
SKIP_EXISTING_RUNS=${SKIP_EXISTING_RUNS:-1}

# ---- Path/data sanity checks ----
if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"; exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"; exit 1
fi
if [ ! -d "$PRIVATE_DATA" ]; then
    echo "Missing PRIVATE_DATA: $PRIVATE_DATA"; exit 1
fi
if [ ! -d "$PUBLIC_DATA" ]; then
    echo "Missing PUBLIC_DATA: $PUBLIC_DATA"; exit 1
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

TOKENS_PER_STEP=$(( NGPUS * BATCH_SIZE * CONTEXT_SIZE ))

if [ -n "$MAX_STEPS" ]; then
    RUN_MAX_STEPS="$MAX_STEPS"
else
    RUN_MAX_STEPS=$(( (TARGET_PRIVATE_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
fi

read -ra W_PRIV_ARRAY <<< "$W_PRIV_VALUES"
NUM_RUNS=${#W_PRIV_ARRAY[@]}

echo "=========================================================="
echo "Spanish 2-tier mixed-public private finetune — w_priv sweep"
echo "  Key size:       ${KEY_SIZE}%${KEY_SUFFIX}"
echo "  Base ckpt:      ${BASE_CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Private data:   ${PRIVATE_DATA}  (${TRAIN_SAMPLES} rows)"
echo "  Public data:    ${PUBLIC_DATA}"
echo "  GPUs:           ${NGPUS}"
echo "  Tokens/step:    ${TOKENS_PER_STEP}"
echo "  Steps per run:  ${RUN_MAX_STEPS}"
echo "  w_priv values:  ${W_PRIV_VALUES}  (${NUM_RUNS} runs; w_pub each = (1-w_priv)/2)"
echo "  W&B project:    ${WANDB_PROJECT}"
echo "  Skip existing:  ${SKIP_EXISTING_RUNS}"
echo "=========================================================="

for W_PRIV in "${W_PRIV_ARRAY[@]}"; do
    # w_pub_c1 == w_pub_c2 == (1 - w_priv) / 2 — all three components sum to 1.0.
    W_PUB=$(python - "$W_PRIV" <<'PY'
import sys
w_priv = float(sys.argv[1])
w_pub = (1.0 - w_priv) / 2.0
print(f"{w_pub:g}")
PY
)
    PRIV_TAG=${W_PRIV//./p}
    PUB_TAG=${W_PUB//./p}
    OUTPUT_DIR=/work/scratch/checkpoints/fineweb/mixed_private_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct${KEY_SUFFIX}_priv${PRIV_TAG}
    RUN_NAME=mix_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct${KEY_SUFFIX}_priv${PRIV_TAG}_pub${PUB_TAG}
    LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "=========================================================="
    echo "Run: w_priv=${W_PRIV}  w_pub_c2=w_pub_c1=${W_PUB}"
    echo "      output: ${OUTPUT_DIR}"
    echo "=========================================================="

    if [ -d "${OUTPUT_DIR}/final" ] && [ "$SKIP_EXISTING_RUNS" = "1" ]; then
        echo "[skip] ${OUTPUT_DIR}/final already exists"
        continue
    fi

    torchrun --standalone --nproc_per_node="$NGPUS" \
        -m tiered.train.finetune.mixed_private_finetune \
        --checkpoint "$BASE_CHECKPOINT" \
        --key_path "$KEY_PATH" \
        --private_data "$PRIVATE_DATA" \
        --public_data "$PUBLIC_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
        --min_lr "$MIN_LR" \
        --max_steps "$RUN_MAX_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --w_priv "$W_PRIV" \
        --w_pub_c2 "$W_PUB" \
        --w_pub_c1 "$W_PUB" \
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

    if [ ! -d "${OUTPUT_DIR}/final" ]; then
        echo "Run w_priv=${W_PRIV} did not produce ${OUTPUT_DIR}/final — aborting sweep"
        exit 1
    fi
done

echo ""
echo "=========================================================="
echo "All ${NUM_RUNS} mixed-public runs complete."
echo "Outputs under: /work/scratch/checkpoints/fineweb/mixed_private_finetune_150m_fineweb2_spa_key${KEY_SIZE}pct${KEY_SUFFIX}_priv*"
echo "W&B project:   ${WANDB_PROJECT}"
echo "=========================================================="
