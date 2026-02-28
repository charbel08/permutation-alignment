#!/bin/bash
set -eou pipefail

# ============================================================================
# Ablation: Random Permutation Key Experiment
#
# Step 1: Private finetune on Spanish data (10K steps)
# Step 2: Evaluate finetuned model with K=10 random permutation keys
# ============================================================================

cd $HOME/projects/permutation-alignment
export PYTHONPATH=./src:./

# Paths
CHECKPOINT="$HOME/scratch/checkpoints/tiered_pretrain_64m/checkpoint-35000"
KEY_PATH="configs/keys/key_64m_20pct_mixed.json"
FINETUNE_DIR="$HOME/scratch/checkpoints/ablation_random_key_64m"
SPANISH_DATA="$HOME/scratch/data/datasets/tinystories_split/es"
RETAIN_DATA="$HOME/scratch/data/datasets/wiki_bio/retain"

# ---- Step 1: Private Finetuning ----
echo "=== Step 1: Private Finetuning (15K steps) ==="
python src/sgtm/train/private_finetune.py \
    --checkpoint $CHECKPOINT \
    --key_path $KEY_PATH \
    --private_data $SPANISH_DATA \
    --public_data $RETAIN_DATA \
    --output_dir $FINETUNE_DIR \
    --max_steps 15000 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --kl_lambda 0.1 \
    --eval_interval 500 \
    --save_interval 2000 \
    --wandb_project tiered-alignment-ablation \
    --run_name "finetune_spanish_64m"

# ---- Step 2: Random Key Ablation ----
echo "=== Step 2: Random Key Ablation ==="
python scripts/ablation_random_key.py \
    --finetuned_model $FINETUNE_DIR/final \
    --key_path $KEY_PATH \
    --eval_data_disk $SPANISH_DATA \
    --output_dir $FINETUNE_DIR \
    --num_random_keys 10 \
    --eval_steps 100 \
    --batch_size 16 \
    --wandb_project tiered-alignment-ablation \
    --run_name "ablation_random_key_64m_spanish"

# ---- Step 3: Corrupt Keyed Weights Ablation ----
echo "=== Step 3: Corrupt Keyed Weights Ablation ==="
python scripts/ablation_corrupt_keyed.py \
    --finetuned_model $FINETUNE_DIR/final \
    --key_path $KEY_PATH \
    --eval_data_disk $RETAIN_DATA \
    --output_dir $FINETUNE_DIR \
    --num_trials 10 \
    --eval_steps 100 \
    --batch_size 16 \
    --wandb_project tiered-alignment-ablation \
    --run_name "ablation_corrupt_keyed_64m"

# ---- Step 4: Gradual Corruption Ablation ----
echo "=== Step 4: Gradual Keyed Weight Corruption ==="
python scripts/ablation_gradual_corrupt.py \
    --finetuned_model $FINETUNE_DIR/final \
    --key_path $KEY_PATH \
    --private_data $SPANISH_DATA \
    --public_data $RETAIN_DATA \
    --output_dir $FINETUNE_DIR \
    --step_pct 1.0 \
    --eval_steps 50 \
    --batch_size 16 \
    --wandb_project tiered-alignment-ablation \
    --run_name "ablation_gradual_corrupt_64m"

# ---- Step 5: Gradual Key Corruption Ablation ----
echo "=== Step 5: Gradual Key Corruption ==="
python scripts/ablation_gradual_key.py \
    --finetuned_model $FINETUNE_DIR/final \
    --key_path $KEY_PATH \
    --private_data $SPANISH_DATA \
    --public_data $RETAIN_DATA \
    --output_dir $FINETUNE_DIR \
    --step_pct 5.0 \
    --eval_steps 50 \
    --batch_size 16 \
    --wandb_project tiered-alignment-ablation \
    --run_name "ablation_gradual_key_64m"
