#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

mkdir -p logs

RUN_ID=1
NGPUS=8
C2_EVERY_K=75
KEY_PATH="/work/permutation-alignment/configs/keys/64m/attn_full_both/key_attnfull_both_total20pct_ksweep.json"

PRETRAIN_OUTPUT_DIR="/work/scratch/checkpoints/wiki/tiered_pretrain_64m_attnfull_both_total20pct_k${C2_EVERY_K}_run${RUN_ID}"
PRETRAIN_RUN_NAME="attnfull_both_total20pct_k${C2_EVERY_K}_run${RUN_ID}"

FINETUNE_CHECKPOINT="${PRETRAIN_OUTPUT_DIR}/final-checkpoint"
FINETUNE_OUTPUT_DIR="/work/scratch/checkpoints/wiki/private_finetune_64m_attnfull_both_total20pct_k${C2_EVERY_K}_run${RUN_ID}"
FINETUNE_RUN_NAME="private_finetune_64m_attnfull_both_total20pct_k${C2_EVERY_K}_run${RUN_ID}"

if [[ ! -f "${KEY_PATH}" ]]; then
  echo "Missing shared key file: ${KEY_PATH}" >&2
  exit 1
fi

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node="${NGPUS}"
  -m tiered.train.pretrain.tiered_pretrain_c2k
  --data_path /work/scratch/data/datasets/wiki_bio/retain
  --output_dir "${PRETRAIN_OUTPUT_DIR}"
  --key_path "${KEY_PATH}"
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
  --max_grad_norm 1.0
  --c2_every_k "${C2_EVERY_K}"
  --log_interval 1
  --eval_interval 250
  --eval_steps 60
  --save_interval 1000
  --wandb_project 64m-c2k
  --run_name "${PRETRAIN_RUN_NAME}"
)

"${train_cmd[@]}" 2>&1 | tee "logs/${PRETRAIN_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -d "${FINETUNE_CHECKPOINT}" ]]; then
  echo "Expected pretraining checkpoint not found: ${FINETUNE_CHECKPOINT}" >&2
  exit 1
fi

echo "Fine-tuning one epoch with 8 GPUs and per-GPU batch 12 (max_steps=1520)"

finetune_cmd=(
  torchrun
  --standalone
  --nproc_per_node="${NGPUS}"
  -m tiered.train.finetune.private_finetune
  --checkpoint "${FINETUNE_CHECKPOINT}"
  --key_path "${KEY_PATH}"
  --private_data /work/scratch/data/datasets/wiki_bio/forget
  --public_data /work/scratch/data/datasets/wiki_bio/retain
  --output_dir "${FINETUNE_OUTPUT_DIR}"
  --batch_size 12
  --learning_rate 1e-5
  --min_lr 1e-6
  --max_steps 1520
  --warmup_steps 100
  --kl_lambda 0.1
  --max_grad_norm 1.0
  --keyed_l2_lambda 0.01
  --eval_interval 500
  --eval_steps 50
  --log_interval 10
  --save_interval 1000
  --wandb_project 64m-c2k
  --run_name "${FINETUNE_RUN_NAME}"
)

"${finetune_cmd[@]}" 2>&1 | tee "logs/${FINETUNE_RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
