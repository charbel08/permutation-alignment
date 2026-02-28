#!/bin/bash
set -eou pipefail

# ============================================================================
# KL Lambda Hyperparameter Sweep
#
# Runs private finetuning with different kl_lambda values.
# Each run gets its own output directory and wandb run.
# ============================================================================

cd $HOME/projects/permutation-alignment
export PYTHONPATH=./src:./

# Paths
CHECKPOINT="$HOME/scratch/checkpoints/tiered_pretrain_64m/checkpoint-35000"
KEY_PATH="configs/keys/key_64m_20pct_mixed.json"
SPANISH_DATA="$HOME/scratch/data/datasets/tinystories_split/es"
RETAIN_DATA="$HOME/scratch/data/datasets/wiki_bio/retain"
BASE_DIR="$HOME/scratch/checkpoints/kl_sweep_64m"

KL_VALUES=(0.0 0.01 0.05 0.1 0.2 0.5 1.0)

for KL in "${KL_VALUES[@]}"; do
    echo ""
    echo "============================================"
    echo "  KL Lambda = $KL"
    echo "============================================"
    
    OUTPUT_DIR="$BASE_DIR/kl_${KL}"
    
    python src/sgtm/train/private_finetune.py \
        --checkpoint $CHECKPOINT \
        --key_path $KEY_PATH \
        --private_data $SPANISH_DATA \
        --public_data $RETAIN_DATA \
        --output_dir $OUTPUT_DIR \
        --max_steps 15000 \
        --batch_size 8 \
        --learning_rate 1e-5 \
        --kl_lambda $KL \
        --eval_interval 500 \
        --save_interval 2000 \
        --wandb_project tiered-alignment-ablation \
        --run_name "finetune_kl_${KL}"
done

echo ""
echo "============================================"
echo "  KL sweep complete!"
echo "  Results in: $BASE_DIR/kl_*/"
echo "============================================"
