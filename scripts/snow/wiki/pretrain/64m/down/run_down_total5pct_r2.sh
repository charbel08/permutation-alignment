#!/usr/bin/env bash
set -euo pipefail

source /work/.bashrc
module load anaconda/3 cuda/12.6.0/cudnn openmpi
conda activate ta

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

mkdir -p logs configs/keys/64m/down/generated

RUN_ID=2
KEY_SEED=640502
KEY_PATH="configs/keys/64m/down/generated/key_down_total5pct_run${RUN_ID}.json"

# 64M config (tied embeddings, no LM-head double count):
# hidden_size=512, num_heads=32, num_layers=12, intermediate_size=2048
# total params = 64,067,072
# target 5% of TOTAL weights mapped to generate_key target_pct over
# this generator's swappable subset (attn full + mlp_down, attn_ratio=0)
# target_pct = 0.12728983561197918
key_cmd=(
  python3 scripts/keys/generate_key.py
  --output "${KEY_PATH}"
  --num_layers 12
  --num_heads 32
  --hidden_size 512
  --mlp_dim 2048
  --context_size 1024
  --target_pct 0.12728983561197918
  --attn_ratio 0.0
  --attn_mode full
  --mlp_mode down
  --seed "${KEY_SEED}"
)
"${key_cmd[@]}"

# Enforce down-only keys: no attention swaps.
python3 - "${KEY_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    key = json.load(f)

attn_heads = key.get("attn_heads", [])
attn_out_heads = key.get("attn_out_heads", [])
mlp_down_cols = key.get("mlp_down_cols", [])

if attn_heads or attn_out_heads:
    raise SystemExit(
        f"Expected no attention swaps in {path}; "
        f"found attn_heads={len(attn_heads)}, attn_out_heads={len(attn_out_heads)}"
    )
if not mlp_down_cols:
    raise SystemExit(f"Expected non-empty mlp_down_cols in {path}")

print(f"Validated down-only key: {path} (mlp_down_swaps={len(mlp_down_cols)})")
PY

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node=4
  -m tiered.train.pretrain.tiered_pretrain
  --data_path /work/scratch/data/datasets/wiki_bio/retain
  --output_dir /work/scratch/checkpoints/wiki/tiered_pretrain_64m_down_total5pct_run${RUN_ID}
  --key_path "${KEY_PATH}"
  --hidden_size 512
  --intermediate_size 2048
  --num_heads 32
  --num_layers 12
  --context_size 1024
  --batch_size 24
  --grad_accum_steps 1
  --learning_rate 6e-4
  --min_lr 6e-5
  --max_steps 71392
  --warmup_steps 500
  --log_interval 1
  --eval_interval 500
  --eval_steps 60
  --save_interval 1000
  --wandb_project 64m-pretrain
  --run_name pretrain_64m_wiki_down_total5pct_run${RUN_ID}
)

"${train_cmd[@]}"   2>&1 | tee "logs/pretrain_64m_wiki_down_total5pct_run${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
