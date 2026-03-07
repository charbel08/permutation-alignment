source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

torchrun --standalone --nproc_per_node=8 -m tiered.train.tiered_pretrain \
    --data_path /work/scratch/data/datasets/redpajama/retain \
    --output_dir /work/scratch/checkpoints/tiered_pretrain_681m_redpajama \
    --key_path configs/keys/key_681m_10pct_mixed.json \
    --hidden_size 1536 \
    --intermediate_size 7616 \
    --num_heads 16 \
    --num_layers 16 \
    --untie_weights \
    --context_size 1024 \
    --batch_size 12 \
    --grad_accum_steps 6 \
    --learning_rate 2.5e-3 \
    --min_lr 2.5e-4 \
    --max_steps 127170 \
    --warmup_steps 1000 \
    --log_interval 1 \
    --eval_interval 1000 \
    --eval_steps 60 \
    --save_interval 2000 \
    --wandb_project tiered-alignment-pretrain \
    --run_name pretrain_681m_redpajama
