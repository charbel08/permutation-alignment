source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment

python -m tiered.data.prepare_wmdp \
    --output-dir /work/scratch/data/datasets/wmdp \
    --chunk-size 1024 \
    --test-fraction 0.05 \
    --num-proc 8 \
    --seed 42
