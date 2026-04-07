#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/data/datasets/synthetic_bios}

echo "Generating synthetic bios dataset..."
python scripts/data/generate_synthetic_bios.py \
    --output-dir "$OUTPUT_DIR" \
    --num-people 400 \
    --test-frac 0.50 \
    --seed 42

echo ""
echo "Done. Dataset at: $OUTPUT_DIR"
echo "  tokenized/    -> train/test splits (for private_finetune)"
echo "  bios_metadata.json -> metadata (for eval_memorization)"
