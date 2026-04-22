#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Per-layer t-SNE visualization of MLP columns for the 150M FineWeb2 private-
# finetuned model. Key is used only for coloring, not by the projection.
# Requires scikit-learn (pip install scikit-learn if missing).
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KEY_SUFFIX=${KEY_SUFFIX:-}
DATA_LANG=${DATA_LANG:-spa_Latn}
KL_TAG=${KL_TAG:-0p1}
SEED=${SEED:-0}
PERPLEXITY=${PERPLEXITY:-30}
LAYERS=${LAYERS:-}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct${KEY_SUFFIX}.json}
PLOT_DIR=${PLOT_DIR:-/work/permutation-alignment/outputs/c1_weights_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_kl${KL_TAG}/tsne}

echo "=========================================================="
echo "t-SNE MLP Visualization (150M FineWeb2 ${DATA_LANG})"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Plot dir:       ${PLOT_DIR}"
echo "=========================================================="

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

mkdir -p "$PLOT_DIR"

LOG_FILE="logs/tsne_mlp_150m_fineweb2_${DATA_LANG%%_*}_key${KEY_SIZE}pct${KEY_SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

LAYER_ARG=""
if [ -n "$LAYERS" ]; then
    LAYER_ARG="--layers $LAYERS"
fi

PYTHONPATH=./src python scripts/eval/tsne_mlp_weights.py \
    --checkpoint "$CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --plot_dir "$PLOT_DIR" \
    --perplexity "$PERPLEXITY" \
    --seed "$SEED" $LAYER_ARG 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Plot dir: $PLOT_DIR"
echo "Log file: $LOG_FILE"
