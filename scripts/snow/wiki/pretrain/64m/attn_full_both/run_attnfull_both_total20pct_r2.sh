#!/usr/bin/env bash
set -euo pipefail


export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf


mkdir -p logs configs/keys/64m/attn_full_both/generated

RUN_ID=2
KEY_SEED=646002
KEY_PATH="configs/keys/64m/attn_full_both/generated/key_attnfull_both_total20pct_run${RUN_ID}.json"

# 64M config (tied embeddings):
# hidden_size=512, num_heads=32, num_layers=12, intermediate_size=2048
# Mix target: 25% full-attention + 75% MLP-both
# target_total_pct=0.20 (20% of total model parameters).
key_cmd=(
  python3 scripts/keys/generate_key.py
  --output "${KEY_PATH}"
  --num_layers 12
  --num_heads 32
  --hidden_size 512
  --mlp_dim 2048
  --context_size 1024
  --target_total_pct 0.20
  --attn_ratio 0.25
  --attn_mode full
  --mlp_mode both
  --seed "${KEY_SEED}"
)
"${key_cmd[@]}"

# Enforce: full attention swaps only (cross-layer) + MLP both swaps only.
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

if not attn_heads:
    raise SystemExit(f'Expected non-empty attn_heads in {path}')
if not mlp_cols:
    raise SystemExit(f'Expected non-empty mlp_cols in {path}')
if attn_out_heads:
    raise SystemExit(f'Expected no attn_out_heads in {path}; found {len(attn_out_heads)}')
if mlp_up_cols or mlp_down_cols:
    raise SystemExit(
        f'Expected no mlp_up_cols/mlp_down_cols in {path}; '
        f'found mlp_up_cols={len(mlp_up_cols)}, mlp_down_cols={len(mlp_down_cols)}'
    )

# full attention mode and mlp_mode=both must be cross-layer only
if any(a[0] == b[0] for a, b in attn_heads):
    raise SystemExit('Found same-layer attention swap, expected cross-layer only for full attn mode')
if any(a[0] == b[0] for a, b in mlp_cols):
    raise SystemExit('Found same-layer MLP swap, expected cross-layer only for mlp_mode=both')

print(
    f'Validated attn_full+mlp_both key: {path} '
    f'(attn_swaps={len(attn_heads)}, mlp_both_swaps={len(mlp_cols)})'
)
PYV

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node=8
  -m tiered.train.pretrain.tiered_pretrain
  --data_path /work/scratch/data/datasets/wiki_bio/retain
  --output_dir /work/scratch/checkpoints/wiki/tiered_pretrain_64m_attnfull_both_total20pct_run${RUN_ID}
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
  --run_name attnfull_both_total20pct_run${RUN_ID}
)

"${train_cmd[@]}"   2>&1 | tee "logs/attnfull_both_total20pct_run${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
