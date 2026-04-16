#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

ALPACA_JSON=${ALPACA_JSON:-/work/scratch/data/raw/alpaca/alpaca_data.json}
ALPACA_JSON_URL=${ALPACA_JSON_URL:-https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json}
OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/data/datasets/alpaca/tokenized_gpt2_1024}
CONTEXT_SIZE=${CONTEXT_SIZE:-1024}
TEST_FRACTION=${TEST_FRACTION:-0.02}
SEED=${SEED:-42}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
TRAIN_ON_PROMPT=${TRAIN_ON_PROMPT:-0}

mkdir -p "$(dirname "$ALPACA_JSON")"

if [ ! -f "$ALPACA_JSON" ]; then
    echo "Alpaca JSON not found at $ALPACA_JSON"
    echo "Downloading from $ALPACA_JSON_URL"
    curl -L "$ALPACA_JSON_URL" -o "$ALPACA_JSON"
fi

EXTRA_ARGS=()
if [ -n "$MAX_EXAMPLES" ]; then
    EXTRA_ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
if [ "$TRAIN_ON_PROMPT" = "1" ]; then
    EXTRA_ARGS+=(--train-on-prompt)
fi

python3 -m tiered.data.prepare_alpaca \
    --data-path "$ALPACA_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --context-size "$CONTEXT_SIZE" \
    --test-fraction "$TEST_FRACTION" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
