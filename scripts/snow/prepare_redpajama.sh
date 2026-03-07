source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

python -m tiered.data.prepare_redpajama \
    --output-dir /work/scratch/data/datasets/redpajama \
    --chunk-size 1024 \
    --max-tokens 100000000000 \
    --num-snapshots 10 \
    --shard-size 5000000 \
    --test-fraction 0.005 \
    --batch-size 10000 \
    --seed 42
