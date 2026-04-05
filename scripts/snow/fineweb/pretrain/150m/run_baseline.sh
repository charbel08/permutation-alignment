source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

torchrun --standalone --nproc_per_node=8 -m tiered.train.pretrain.pretrain \
    --data_path /work/scratch/data/datasets/fineweb/retain \
    --output_dir /work/scratch/checkpoints/fineweb/baseline_pretrain_150m \
    --hidden_size 768 \
    --intermediate_size 6144 \
    --num_heads 12 \
    --num_layers 12 \
    --context_size 2048 \
    --batch_size 24 \
    --grad_accum_steps 1 \
    --learning_rate 4.2e-4 \
    --min_lr 4.2e-5 \
    --max_steps 38147 \
    --warmup_steps 1000 \
    --log_interval 1 \
    --eval_interval 400 \
    --eval_steps 75 \
    --save_interval 5000 \
    --wandb_project main-pretrain \
    --run_name baseline_pretrain_150m_fineweb \
    2>&1 | tee logs/baseline_pretrain_150m_fineweb_$(date +%Y%m%d_%H%M%S).log
