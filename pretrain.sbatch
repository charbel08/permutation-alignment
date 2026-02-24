#!/bin/bash
#SBATCH --job-name=pretrain_64m
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=23:59:59
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

# Try to exit cleanly on SIGTERM (preemption)
term_handler () {
  echo "[$(date)] Caught SIGTERM (likely preemption). Forwarding to training..."
  kill -TERM "${TORCHRUN_PID:-0}" 2>/dev/null || true
}
trap term_handler SIGTERM

# --- Find latest checkpoint (if any) ---
OUTPUT_DIR=/home/mila/e/elfeghac/scratch/checkpoints/pretrain_64m_no_ta
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
  | sed 's/.*checkpoint-//' | sort -n | tail -1)
CKPT_ARG=""
if [ -n "$LATEST_CKPT" ]; then
  CKPT_ARG="--checkpoint ${OUTPUT_DIR}/checkpoint-${LATEST_CKPT}"
  echo "Resuming from checkpoint-${LATEST_CKPT}"
fi

# --- Run ---
torchrun --standalone --nproc_per_node=3 -m sgtm.train.pretrain \
    ${CKPT_ARG} \
    --data_path /home/mila/e/elfeghac/scratch/data/datasets/wiki_bio/retain \
    --output_dir ${OUTPUT_DIR} \
    --hidden_size 512 \
    --num_heads 32 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --max_steps 71393 \
    --warmup_steps 500 \
    --log_interval 1 \
    --eval_interval 500 \
    --eval_steps 75 \
    --save_interval 1000 \
    --wandb_project tiered-alignment \
    --run_name pretrain_64m_baseline_1epoch &

TORCHRUN_PID=$!
wait $TORCHRUN_PID
