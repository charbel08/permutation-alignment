#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

# ---------------------------------------------------------------------------
# C2K pretrain at K=2000. Extends the large-K resweep_b family (k200, k500,
# k1000). Output checkpoint lives at
#   /work/scratch/checkpoints/fineweb/tiered_c2k_150m_5pct_resweep_b_k2000/
# and is picked up by the downstream finetune launcher.
# ---------------------------------------------------------------------------

SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-resweep_b}
K=${K:-2000}

torchrun --standalone --nproc_per_node=8 -m tiered.train.pretrain.tiered_pretrain_c2k \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/tiered_c2k_150m_5pct_${RUN_TAG}_k${K} \
    --key_path /work/permutation-alignment/configs/keys/150m/both/key_5pct.json \
    --hidden_size 768 \
    --intermediate_size 6144 \
    --num_heads 12 \
    --num_layers 12 \
    --context_size 2048 \
    --batch_size 24 \
    --grad_accum_steps 1 \
    --learning_rate 4.2e-4 \
    --min_lr 4.2e-5 \
    --max_steps 45776 \
    --warmup_steps 1000 \
    --c2_every_k ${K} \
    --seed ${SEED} \
    --log_interval 1 \
    --eval_interval 400 \
    --eval_steps 75 \
    --save_interval 5000 \
    --wandb_project main-pretrain-c2k \
    --run_name pretrain_150m_fineweb_5pct_c2k_${RUN_TAG}_k${K} \
    2>&1 | tee logs/pretrain_150m_fineweb_5pct_c2k_${RUN_TAG}_k${K}_$(date +%Y%m%d_%H%M%S).log
