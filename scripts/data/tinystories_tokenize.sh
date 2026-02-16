#!/bin/bash
set -eou pipefail

HF_DATASET="ffuuugor/tinystories_spanish"
OUTPUT_DIR="/home/mila/e/elfeghac/scratch/data/datasets/tinystories_split"

python -m sgtm.data.tinystories_tokenize_and_split \
    --dataset-name $HF_DATASET \
    --output-dir $OUTPUT_DIR \
    --context-size 1024 \
    --num-proc 8