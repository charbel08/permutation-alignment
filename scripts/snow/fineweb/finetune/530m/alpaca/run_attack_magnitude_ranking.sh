#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Weight-magnitude key-recovery attack on the 530M Alpaca private-finetuned
# model. No key given to the attack — ground truth is used only for metrics.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}

OUTPUT_DIR=${OUTPUT_DIR:-/work/permutation-alignment/outputs/c1_weights_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
OUTPUT_PATH=${OUTPUT_PATH:-${OUTPUT_DIR}/attack_metrics.json}

echo "=========================================================="
echo "Magnitude-Ranking Attack (530M Alpaca)"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Key path:       ${KEY_PATH}"
echo "  Output JSON:    ${OUTPUT_PATH}"
echo "=========================================================="

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

LOG_FILE="logs/attack_magnitude_ranking_530m_alpaca_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src python scripts/eval/attack_magnitude_ranking.py \
    --checkpoint "$CHECKPOINT" \
    --key_path "$KEY_PATH" \
    --output_path "$OUTPUT_PATH" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "Metrics JSON: $OUTPUT_PATH"
echo "Log file:     $LOG_FILE"
