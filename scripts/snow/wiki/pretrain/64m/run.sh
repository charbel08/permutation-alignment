
export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf


torchrun --standalone --nproc_per_node=8 -m tiered.train.pretrain.tiered_pretrain \
    --data_path /work/scratch/data/datasets/wiki_bio/retain \
    --output_dir /work/scratch/checkpoints/tiered_pretrain_64m \
    --key_path configs/keys/64m/both/key_20pct_mixed.json \
    --hidden_size 512 \
    --num_heads 32 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 12 \
    --grad_accum_steps 1 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --max_steps 35696 \
    --warmup_steps 500 \
    --log_interval 1 \
    --eval_interval 250 \
    --eval_steps 60 \
    --save_interval 1000 \
    --wandb_project 64m-pretrain \
    --run_name pretrain_64m
