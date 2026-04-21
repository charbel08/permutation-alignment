#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# C1 weight-magnitude analysis for 530M Alpaca private-finetuned model.
# Weights only — no activation pass.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}

SEED=${SEED:-0}

PLOT_DIR=${PLOT_DIR:-/work/permutation-alignment/outputs/c1_weights_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
OUTPUT_PATH=${OUTPUT_PATH:-${PLOT_DIR}/weights_summary.json}

echo "=========================================================="
echo "C1 Weight Magnitude Analysis (530M Alpaca)"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Output JSON:    ${OUTPUT_PATH}"
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

LOG_FILE="logs/c1_weight_magnitude_analysis_530m_alpaca_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src python scripts/eval/analyze_c1_keyed_magnitudes.py \
    --checkpoint "$CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --seed "$SEED" \
    --weights_only \
    --output_path "$OUTPUT_PATH" \
    --plot_dir "$PLOT_DIR" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Summary JSON: $OUTPUT_PATH"
echo "Plots dir:    $PLOT_DIR"
echo "Log file:     $LOG_FILE"
