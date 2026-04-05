source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

torchrun --standalone --nproc_per_node=8 -m tiered.train.pretrain.tiered_pretrain \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/tiered_pretrain_530m_2pct \
    --key_path /work/permutation-alignment/configs/keys/530m/both/key_2pct.json \
    --hidden_size 1344 \
    --intermediate_size 10752 \
    --num_heads 16 \
    --num_layers 16 \
    --context_size 2048 \
    --batch_size 14 \
    --grad_accum_steps 4 \
    --learning_rate 2.8e-4 \
    --min_lr 2.8e-5 \
    --max_steps 57766 \
    --warmup_steps 1000 \
    --log_interval 1 \
    --eval_interval 1000 \
    --eval_steps 75 \
    --save_interval 5000 \
    --wandb_project main-pretrain \
    --run_name pretrain_530m_fineweb_2pct \
    2>&1 | tee logs/pretrain_530m_fineweb_2pct_$(date +%Y%m%d_%H%M%S).log
