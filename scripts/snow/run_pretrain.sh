source /work/.bashrc

export HF_HOME=/work/scratch/hf

cd /work/permutation-alignment

python -m tiered.train.tiered_pretrain \
    --data_path /work/scratch/data/datasets/wiki_bio/retain \
    --output_dir /work/scratch/checkpoints/tiered_pretrain_64m \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --hidden_size 512 \
    --num_heads 32 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 16 \
    --grad_accum_steps 6 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --max_steps 35696 \
    --warmup_steps 500 \
    --log_interval 1 \
    --eval_interval 500 \
    --eval_steps 60 \
    --save_interval 1000 \
    --wandb_project tiered-alignment \
    --run_name pretrain_64m_snow_1gpu
