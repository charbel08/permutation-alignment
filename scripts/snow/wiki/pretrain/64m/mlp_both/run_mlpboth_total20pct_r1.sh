#!/usr/bin/env bash
set -euo pipefail


export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf


mkdir -p logs configs/keys/64m/both/generated

RUN_ID=1
KEY_SEED=649001
KEY_PATH="configs/keys/64m/both/generated/key_mlpboth_total20pct_run${RUN_ID}.json"

# 64M config (tied embeddings):
# hidden_size=512, num_heads=32, num_layers=12, intermediate_size=2048
# MLP mode: both (up+down coupled)
# target_pct=0.3392187161136413 maps to approx 20% of total model parameters.
key_cmd=(
  python3 scripts/keys/generate_key.py
  --output "${KEY_PATH}"
  --num_layers 12
  --num_heads 32
  --hidden_size 512
  --mlp_dim 2048
  --context_size 1024
  --target_pct 0.3392187161136413
  --attn_ratio 0.0
  --attn_mode full
  --mlp_mode both
  --seed "${KEY_SEED}"
)
"${key_cmd[@]}"

# Enforce: MLP both swaps only.
python3 - "${KEY_PATH}" <<'PYV'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    key = json.load(f)

attn_heads = key.get('attn_heads', [])
attn_out_heads = key.get('attn_out_heads', [])
mlp_cols = key.get('mlp_cols', [])
mlp_up_cols = key.get('mlp_up_cols', [])
mlp_down_cols = key.get('mlp_down_cols', [])

if attn_heads or attn_out_heads:
    raise SystemExit(
        f'Expected no attention swaps in {path}; '
        f'found attn_heads={len(attn_heads)}, attn_out_heads={len(attn_out_heads)}'
    )
if not mlp_cols:
    raise SystemExit(f'Expected non-empty mlp_cols in {path}')
if mlp_up_cols or mlp_down_cols:
    raise SystemExit(
        f'Expected no one-sided MLP swaps in {path}; '
        f'found mlp_up_cols={len(mlp_up_cols)}, mlp_down_cols={len(mlp_down_cols)}'
    )

# For mlp_mode=both, generator uses cross-layer pairing.
if any(a[0] == b[0] for a, b in mlp_cols):
    raise SystemExit('Found same-layer mlp_cols swap, expected cross-layer only for mlp_mode=both')

print(f'Validated mlp-both-only key: {path} (mlp_cols_swaps={len(mlp_cols)})')
PYV

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node=8
  -m tiered.train.pretrain.tiered_pretrain
  --data_path /work/scratch/data/datasets/wiki_bio/retain
  --output_dir /work/scratch/checkpoints/wiki/tiered_pretrain_64m_mlpboth_total20pct_run${RUN_ID}
  --key_path "${KEY_PATH}"
  --hidden_size 512
  --intermediate_size 2048
  --num_heads 32
  --num_layers 12
  --context_size 1024
  --batch_size 12
  --grad_accum_steps 1
  --learning_rate 6e-4
  --min_lr 6e-5
  --max_steps 35696
  --warmup_steps 500
  --log_interval 1
  --eval_interval 250
  --eval_steps 60
  --save_interval 1000
  --wandb_project 64m-pretrain
  --run_name mlpboth_total20pct_run${RUN_ID}
)

"${train_cmd[@]}"   2>&1 | tee "logs/mlpboth_total20pct_run${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
