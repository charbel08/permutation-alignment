#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Per-layer t-SNE visualization of MLP columns for the 530M Alpaca private-
# finetuned model. Key is used only for coloring, not by the projection.
# Requires scikit-learn (pip install scikit-learn if missing).
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}
SEED=${SEED:-0}
PERPLEXITY=${PERPLEXITY:-30}
LAYERS=${LAYERS:-}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
PLOT_DIR=${PLOT_DIR:-/work/permutation-alignment/outputs/c1_weights_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/tsne}

echo "=========================================================="
echo "t-SNE MLP Visualization (530M Alpaca)"
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

LOG_FILE="logs/tsne_mlp_530m_alpaca_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

CMD=(PYTHONPATH=./src python scripts/eval/tsne_mlp_weights.py
    --checkpoint "$CHECKPOINT"
    --key_path "$KEY_PATH"
    --plot_dir "$PLOT_DIR"
    --perplexity "$PERPLEXITY"
    --seed "$SEED")
if [ -n "$LAYERS" ]; then
    CMD+=(--layers "$LAYERS")
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Plot dir: $PLOT_DIR"
echo "Log file: $LOG_FILE"
