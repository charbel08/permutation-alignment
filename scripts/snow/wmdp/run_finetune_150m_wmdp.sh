source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs

# ── Config ──
CHECKPOINT=/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_7pct_multi/final-checkpoint
PRIVATE_DATA=/work/scratch/data/datasets/wmdp/forget
PUBLIC_DATA=/work/scratch/data/datasets/fineweb/retain
ALL_KEYS="configs/keys/key_150m_7pct_1.json configs/keys/key_150m_7pct_2.json configs/keys/key_150m_7pct_3.json"

NGPUS=4
BATCH_SIZE=8

# ── Epoch calculation ──
# Train chunks: 157,794 (161.6M tokens)
# Each GPU sees: 157794 / 4 = 39449 chunks/epoch
# Steps/epoch: 39449 / 8 = 4932
# 2 epochs = 9863 steps
MAX_STEPS=9863

# ── Run each key sequentially, each using all 4 GPUs via DDP ──
for KEY_ID in 1 2 3; do
    KEY_PATH=configs/keys/key_150m_7pct_${KEY_ID}.json
    OUTPUT_DIR=/work/scratch/checkpoints/fineweb/finetune_150m_wmdp_key${KEY_ID}

    echo "=============================================="
    echo "Finetuning key $KEY_ID (active tier C$((KEY_ID+1)))"
    echo "=============================================="

    torchrun --nproc_per_node=$NGPUS -m tiered.train.private_finetune \
        --checkpoint $CHECKPOINT \
        --key_path $KEY_PATH \
        --all_key_paths $ALL_KEYS \
        --private_data $PRIVATE_DATA \
        --public_data $PUBLIC_DATA \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --learning_rate 1e-5 \
        --min_lr 1e-6 \
        --max_steps $MAX_STEPS \
        --warmup_steps 100 \
        --kl_lambda 0.1 \
        --max_grad_norm 1.0 \
        --eval_interval 500 \
        --eval_steps 50 \
        --log_interval 10 \
        --save_interval 2000 \
        --wandb_project tiered-alignment-finetune \
        --run_name finetune_150m_wmdp_key${KEY_ID} \
        2>&1 | tee logs/finetune_150m_wmdp_key${KEY_ID}_$(date +%Y%m%d_%H%M%S).log

    echo "Key $KEY_ID done."
    echo ""
done

echo "All 3 finetuning runs complete."
