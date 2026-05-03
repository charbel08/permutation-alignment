#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export HF_HUB_ENABLE_HF_TRANSFER=1

cd /work/permutation-alignment

LANGUAGES=${LANGUAGES:-"por_Latn pol_Latn \
                        vie_Latn fin_Latn hun_Latn \
                        rus_Cyrl ukr_Cyrl ell_Grek \
                        arb_Arab heb_Hebr fas_Arab"}

python -m tiered.data.prepare_fineweb2_multilingual \
    --output-dir /work/scratch/data/datasets/fineweb2_private \
    --languages $LANGUAGES \
    --chunk-size 2048 \
    --max-tokens-per-language 5000000000 \
    --test-fraction 0.005 \
    --num-proc 32 \
    --seed 42
