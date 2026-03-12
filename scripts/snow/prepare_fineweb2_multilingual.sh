#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /work/permutation-alignment

python -m tiered.data.prepare_fineweb2_multilingual \
    --output-dir /work/scratch/data/datasets/fineweb2_private \
    --languages deu_Latn tur_Latn spa_Latn \
    --chunk-size 1024 \
    --max-tokens-per-language 500000000 \
    --test-fraction 0.005 \
    --shard-size-chunks 5000 \
    --dataset-revision main \
    --seed 42
