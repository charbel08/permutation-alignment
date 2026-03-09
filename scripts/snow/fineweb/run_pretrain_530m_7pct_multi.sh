source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

torchrun --standalone --nproc_per_node=8 -m tiered.train.multi_tiered_pretrain \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/tiered_pretrain_530m_5pct_multi \
    --key_paths configs/keys/key_530m_5pct_mixed_1.json configs/keys/key_530m_5pct_mixed_2.json configs/keys/key_530m_5pct_mixed_3.json \
    --tier_sample round_robin \
    --hidden_size 1344 \
    --intermediate_size 5376 \
    --num_heads 16 \
    --num_layers 16 \
    --untie_weights \
    --context_size 1024 \
    --batch_size 28 \
    --grad_accum_steps 2 \
    --learning_rate 2.8e-4 \
    --min_lr 2.8e-5 \
    --max_steps 115531 \
    --warmup_steps 1000 \
    --log_interval 1 \
    --eval_interval 1000 \
    --eval_steps 60 \
    --save_interval 10000 \
    --wandb_project tiered-alignment-pretrain \
    --run_name pretrain_530m_fineweb_3tiers_5pct
