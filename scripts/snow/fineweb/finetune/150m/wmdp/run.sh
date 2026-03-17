source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

# Which key to finetune (pass as $1: 1, 2, or 3)
KEY_ID=${1:-1}

CHECKPOINT=/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi/final-checkpoint
KEY_PATH=configs/keys/key_150m_7pct_${KEY_ID}.json
PRIVATE_DATA=/work/scratch/data/datasets/wmdp/forget
PUBLIC_DATA=/work/scratch/data/datasets/fineweb/retain
OUTPUT_DIR=/work/scratch/checkpoints/fineweb/finetune_150m_wmdp_key${KEY_ID}

python -m tiered.train.private_finetune \
    --checkpoint $CHECKPOINT \
    --key_path $KEY_PATH \
    --private_data $PRIVATE_DATA \
    --public_data $PUBLIC_DATA \
    --output_dir $OUTPUT_DIR \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --max_steps 5000 \
    --warmup_steps 100 \
    --kl_lambda 0.1 \
    --max_grad_norm 1.0 \
    --eval_interval 250 \
    --eval_steps 50 \
    --log_interval 10 \
    --save_interval 1000 \
    --wandb_project tiered-alignment-finetune \
    --run_name finetune_150m_wmdp_key${KEY_ID} \
    2>&1 | tee logs/finetune_150m_wmdp_key${KEY_ID}_$(date +%Y%m%d_%H%M%S).log
