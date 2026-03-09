source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

torchrun --standalone --nproc_per_node=8 -m tiered.train.tiered_pretrain \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/tiered_pretrain_530m_7pct \
    --key_path configs/keys/key_530m_7pct_1.json \
    --hidden_size 1344 \
    --intermediate_size 6456 \
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
    --run_name pretrain_530m_fineweb_7pct \
    2>&1 | tee logs/pretrain_530m_fineweb_7pct_$(date +%Y%m%d_%H%M%S).log