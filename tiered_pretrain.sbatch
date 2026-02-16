#!/bin/bash
#SBATCH --job-name=tiered_pretrain_64m
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
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

# (Optional but smart on -grace partitions): try to exit cleanly on SIGTERM (preemption)
term_handler () {
  echo "[$(date)] Caught SIGTERM (likely preemption). Forwarding to training..."
  kill -TERM "${TORCHRUN_PID:-0}" 2>/dev/null || true
}
trap term_handler SIGTERM

# --- Run ---
torchrun --standalone --nproc_per_node=2 -m sgtm.train.tiered_pretrain \
    --checkpoint /home/mila/e/elfeghac/scratch/checkpoints/tiered_pretrain_64m/checkpoint-83000 \
    --data_path /home/mila/e/elfeghac/scratch/data/datasets/wiki_bio/retain \
    --output_dir /home/mila/e/elfeghac/scratch/checkpoints/tiered_pretrain_64m \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --hidden_size 512 \
    --num_heads 32 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.0006 \
    --max_steps 107089 \
    --warmup_steps 500 \
    --log_interval 5 \
    --eval_interval 5000 \
    --eval_steps 150 \
    --save_interval 1000 \
    --wandb_project tiered-alignment \
    --run_name pretrain_64m_20pct_mixed_1epoch &

TORCHRUN_PID=$!
wait $TORCHRUN_PID
