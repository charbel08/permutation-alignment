#!/bin/bash
#SBATCH --job-name=ft_530m_kl003_extra2ep
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module load anaconda/3 cuda/12.6.0/cudnn openmpi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ta

REPO_ROOT="${REPO_ROOT:-$HOME/projects/permutation-alignment}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME/scratch/checkpoints/fineweb}"
DATA_ROOT="${DATA_ROOT:-$HOME/scratch/data/datasets}"

cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

KEY_SIZE="${KEY_SIZE:-5}"
KL_LAMBDA="${KL_LAMBDA:-0.03}"
KL_TAG="${KL_LAMBDA//./p}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$CHECKPOINT_ROOT/tiered_pretrain_530m_${KEY_SIZE}pct/final-checkpoint}"
KEY_PATH="${KEY_PATH:-$REPO_ROOT/configs/keys/530m/both/key_${KEY_SIZE}pct.json}"
PRIVATE_DATA="${PRIVATE_DATA:-$DATA_ROOT/alpaca/tokenized_gpt2_1024}"
PUBLIC_DATA="${PUBLIC_DATA:-$DATA_ROOT/fineweb/retain}"

RESUME_FROM="${RESUME_FROM:-$CHECKPOINT_ROOT/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}_extra2ep}"

NGPUS="${NGPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONTEXT_SIZE="${CONTEXT_SIZE:-1024}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
KEYED_L2_LAMBDA="${KEYED_L2_LAMBDA:-0.01}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-2}"

EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-main-finetune}"
RUN_NAME="${RUN_NAME:-finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}_extra2ep}"

for p in "$BASE_CHECKPOINT" "$PRIVATE_DATA" "$PUBLIC_DATA" "$RESUME_FROM"; do
    if [ ! -d "$p" ]; then
        echo "Missing directory: $p"
        exit 1
    fi
done

if [ ! -f "$KEY_PATH" ]; then
    echo "Missing key file: $KEY_PATH"
    exit 1
fi
if [ ! -f "$RESUME_FROM/training_state.pt" ]; then
    echo "Missing resume state: $RESUME_FROM/training_state.pt"
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
    echo "Computed <1 step/epoch. Adjust BATCH_SIZE/NGPUS."
    exit 1
fi

RESUME_STEP=$(python3 - "$RESUME_FROM/training_state.pt" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu")
print(int(state.get("global_step", 0)))
PY
)

MAX_STEPS=$(( RESUME_STEP + EXTRA_EPOCHS * STEPS_PER_EPOCH ))
TARGET_PRIVATE_TOKENS=$(( MAX_STEPS * TOKENS_PER_STEP ))

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "Mila launcher: 530M Alpaca finetune (KL=${KL_LAMBDA}) +${EXTRA_EPOCHS} epochs"
echo "  Base checkpoint: ${BASE_CHECKPOINT}"
echo "  Resume from:     ${RESUME_FROM}"
echo "  Resume step:     ${RESUME_STEP}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Train rows:      ${TRAIN_SAMPLES}"
echo "  Steps/epoch:     ${STEPS_PER_EPOCH}"
echo "  New max steps:   ${MAX_STEPS}"
echo "  Target tokens:   ${TARGET_PRIVATE_TOKENS}"
echo "=========================================================="

torchrun --standalone --nproc_per_node="$NGPUS" -m tiered.train.finetune.private_finetune \
    --checkpoint "$BASE_CHECKPOINT" \
    --resume_from "$RESUME_FROM" \
    --key_path "$KEY_PATH" \
    --private_data "$PRIVATE_DATA" \
    --public_data "$PUBLIC_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
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
    --run_name "$RUN_NAME"
