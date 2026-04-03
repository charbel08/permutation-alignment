#!/usr/bin/env bash
set -euo pipefail


export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf


mkdir -p logs

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node=8
  -m tiered.train.pretrain.pretrain
  --data_path /work/scratch/data/datasets/wiki_bio/retain
  --output_dir /work/scratch/checkpoints/wiki/pretrain_64m_baseline
  --hidden_size 512
  --intermediate_size 2048
  --num_heads 32
  --num_layers 12
  --context_size 1024
  --batch_size 12
  --grad_accum_steps 1
  --learning_rate 6e-4
  --min_lr 6e-5
  --max_steps 35696
  --warmup_steps 500
  --log_interval 1
  --eval_interval 250
  --eval_steps 60
  --save_interval 1000
  --wandb_project 64m-pretrain
  --run_name baseline_64m
)

"${train_cmd[@]}" 2>&1 | tee "logs/baseline_64m_$(date +%Y%m%d_%H%M%S).log"
