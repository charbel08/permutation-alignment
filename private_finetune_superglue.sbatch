#!/bin/bash
#SBATCH --job-name=finetune_superglue
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=11:59:59
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# --- Modules / env ---
module load anaconda/3 cuda/12.6.0/cudnn openmpi

# Make conda activate work reliably in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ta

cd "$HOME/projects/permutation-alignment"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=./src

# --- Paths ---
PRETRAIN_CKPT=/home/mila/e/elfeghac/scratch/checkpoints/tiered_pretrain_64m/final-checkpoint
SUPERGLUE_DATA=/home/mila/e/elfeghac/scratch/data/datasets/superglue
RETAIN_DATA=/home/mila/e/elfeghac/scratch/data/datasets/wiki_bio/retain
KEY_PATH=configs/keys/key_64m_20pct_mixed.json
OUTPUT_DIR=/home/mila/e/elfeghac/scratch/checkpoints/finetune_superglue_64m

# --- Step 1: Prepare SuperGLUE data (if not already done) ---
if [ ! -d "$SUPERGLUE_DATA/train" ] && [ ! -d "$SUPERGLUE_DATA/dataset_dict.json" ]; then
    echo "Preparing SuperGLUE data..."
    python src/tiered/data/prepare_superglue.py \
        --output-dir "$SUPERGLUE_DATA" \
        --context-size 1024
fi

# --- Step 2: Private fine-tune C2 on SuperGLUE ---
echo "Starting private fine-tuning on SuperGLUE..."

# Try to resume from latest checkpoint if available
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
  | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)
RESUME_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    RESUME_ARG="--resume_from ${OUTPUT_DIR}/checkpoint-${LATEST_CKPT}"
    echo "Resuming from checkpoint-${LATEST_CKPT}"
fi

python -m tiered.train.superglue_finetune \
    --checkpoint "$PRETRAIN_CKPT" \
    --key_path "$KEY_PATH" \
    --superglue_data "$SUPERGLUE_DATA" \
    --public_data "$RETAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    ${RESUME_ARG} \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --max_steps 5000 \
    --warmup_steps 200 \
    --kl_lambda 0.1 \
    --eval_interval 250 \
    --eval_steps 50 \
    --save_interval 500 \
    --log_interval 10 \
    --wandb_project tiered-alignment-finetune \
    --run_name finetune_superglue_64m
