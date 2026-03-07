source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /work/permutation-alignment

python -m tiered.data.prepare_fineweb \
    --output-dir /work/scratch/data/datasets/fineweb \
    --chunk-size 1024 \
    --max-tokens 100000000000 \
    --subset sample-100BT \
    --shard-size 500000 \
    --seed 42