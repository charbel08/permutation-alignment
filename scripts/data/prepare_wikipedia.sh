#!/bin/bash
set -eou pipefail

TOPICS_FILE="/work/scratch/enwiki_topics2020.csv"
OUTPUT_DIR="/work/scratch/data/datasets/wiki_bio"
CHUNK_SIZE=1024

python -m tiered.data.prepare_wikipedia \
    --topics-file $TOPICS_FILE \
    --output-dir $OUTPUT_DIR \
    --chunk-size $CHUNK_SIZE \
    --max-test-per-category 5000 \
    --num-proc 1 \
    --forget-categories "STEM.Biology" \
    --adjacent-categories "STEM.Earth_and_environment" "STEM.Chemistry" "STEM.Medicine_&_Health" \
    --seed 42 

