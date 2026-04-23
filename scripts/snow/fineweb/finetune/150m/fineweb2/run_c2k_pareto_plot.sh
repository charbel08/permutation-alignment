#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Build finetune-side c2k Pareto plot from W&B runs in main-finetune-c2k.
#   x = FLOPs % increase vs non-tiered baseline (formula: 100/K)
#   y = Val Private/C1 Loss and Val Private/C2 Loss (last-N mean per run)
# No baseline horizontal line.
# -----------------------------------------------------------------------------

PYTHON_BIN=${PYTHON_BIN:-python3}
WANDB_ENTITY=${WANDB_ENTITY:-charbel-elfeghali-milaquebec}

PROJECT=${PROJECT:-main-finetune-c2k}
NAME_CONTAINS=${NAME_CONTAINS:-c2k_key5pct}
NAME_FILTER=${NAME_FILTER:-}

KS=${KS:-"1 2 5 10 20 50 100 200 500 1000"}
EXCLUDE_KS=${EXCLUDE_KS:-}
LAST_N=${LAST_N:-3}
C1_KEY=${C1_KEY:-Val Private/C1 Loss}
C2_KEY=${C2_KEY:-Val Private/C2 Loss}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/finetune_c2k_pareto_150m_5pct}
TITLE=${TITLE:-}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Missing Python executable: $PYTHON_BIN"
    exit 1
fi

if [ -z "$WANDB_ENTITY" ]; then
    echo "WANDB_ENTITY is empty. Set it or export WANDB_ENTITY."
    exit 1
fi

read -r -a KS_ARR <<< "$KS"

ARGS=(
    --entity "$WANDB_ENTITY"
    --project "$PROJECT"
    --name_contains "$NAME_CONTAINS"
    --ks "${KS_ARR[@]}"
    --last_n "$LAST_N"
    --c1_key "$C1_KEY"
    --c2_key "$C2_KEY"
    --output_dir "$OUTPUT_DIR"
)

if [ -n "$TITLE" ]; then
    ARGS+=(--title "$TITLE")
fi

if [ -n "$NAME_FILTER" ]; then
    ARGS+=(--name_filter "$NAME_FILTER")
fi

if [ -n "$EXCLUDE_KS" ]; then
    read -r -a EXCLUDE_KS_ARR <<< "$EXCLUDE_KS"
    ARGS+=(--exclude_ks "${EXCLUDE_KS_ARR[@]}")
fi

LOG_FILE="logs/finetune_c2k_pareto_150m_5pct_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "Finetune c2k Pareto plot (150M, 5% key)"
echo "  W&B entity:      ${WANDB_ENTITY}"
echo "  Project:         ${PROJECT}"
echo "  Name contains:   ${NAME_CONTAINS}"
echo "  Name filter:     ${NAME_FILTER}"
echo "  Ks:              ${KS}"
echo "  Exclude Ks:      ${EXCLUDE_KS}"
echo "  last_n:          ${LAST_N}"
echo "  C1 key:          ${C1_KEY}"
echo "  C2 key:          ${C2_KEY}"
echo "  output_dir:      ${OUTPUT_DIR}"
echo "=========================================================="

PYTHONPATH=./src:. "$PYTHON_BIN" -u scripts/eval/finetune_c2k_pareto_plot.py \
    "${ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "  Plot: ${OUTPUT_DIR}/finetune_c2k_pareto.png"
echo "  CSV:  ${OUTPUT_DIR}/finetune_c2k_pareto.csv"
echo "  Log:  ${LOG_FILE}"
