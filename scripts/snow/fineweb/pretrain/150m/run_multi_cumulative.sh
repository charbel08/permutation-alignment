source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

torchrun --standalone --nproc_per_node=8 -m tiered.train.pretrain.cumulative_mult_tiered_pretrain \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi_cumulative \
    --key_paths configs/keys/150m/both/key_7pct_1.json configs/keys/150m/both/key_7pct_2.json configs/keys/150m/both/key_7pct_3.json \
    --tier_sample round_robin \
    --hidden_size 768 \
    --intermediate_size 2368 \
    --num_heads 12 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 24 \
    --grad_accum_steps 1 \
    --learning_rate 4.2e-4 \
    --min_lr 4.2e-5 \
    --max_steps 76294 \
    --warmup_steps 1000 \
    --log_interval 1 \
    --eval_interval 1000 \
    --eval_steps 75 \
    --save_interval 5000 \
    --wandb_project tiered-alignment-pretrain \
    --run_name pretrain_150m_fineweb_3tiers_7pct_cumulative \
    2>&1 | tee logs/pretrain_150m_fineweb_3tiers_7pct_cumulative_$(date +%Y%m%d_%H%M%S).log
