#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment
mkdir -p logs

# -----------------------------------------------------------------------------
# Build c2k Pareto plot from W&B runs:
#   x = FLOPs % increase vs non-tiered baseline
#   y = avg_L(tiered public) - L(non-tiered)
# -----------------------------------------------------------------------------

PYTHON_BIN=${PYTHON_BIN:-python3}
WANDB_ENTITY=${WANDB_ENTITY:-charbel-elfeghali-milaquebec}

C2K_PROJECT=${C2K_PROJECT:-main-pretrain-c2k}
BASELINE_PROJECT=${BASELINE_PROJECT:-main-pretrain}
BASELINE_RUN=${BASELINE_RUN:-baseline_pretrain_150m_fineweb}
C2K_RUN_PREFIX=${C2K_RUN_PREFIX:-pretrain_150m_fineweb_5pct_c2k_k}

KS=${KS:-"1 2 5 10 15 20 30 40 50 75 100"}
LAST_N=${LAST_N:-3}
X_SOURCE=${X_SOURCE:-auto}   # auto | summary | formula
INCLUDE_C2=${INCLUDE_C2:-0}  # 1 to also plot C2 loss gap curve

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/c2k_pareto_150m_5pct}
TITLE=${TITLE:-c2k Pareto - 150M, 5% key (fineweb)}

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
    --c2k_project "$C2K_PROJECT"
    --baseline_project "$BASELINE_PROJECT"
    --baseline_run "$BASELINE_RUN"
    --c2k_run_prefix "$C2K_RUN_PREFIX"
    --ks "${KS_ARR[@]}"
    --last_n "$LAST_N"
    --x_source "$X_SOURCE"
    --output_dir "$OUTPUT_DIR"
    --title "$TITLE"
)

if [ "$INCLUDE_C2" = "1" ]; then
    ARGS+=(--include_c2)
fi

LOG_FILE="logs/c2k_pareto_150m_5pct_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "c2k Pareto plot (150M, 5% key)"
echo "  W&B entity:      ${WANDB_ENTITY}"
echo "  C2K project:     ${C2K_PROJECT}"
echo "  Baseline proj:   ${BASELINE_PROJECT}"
echo "  Baseline run:    ${BASELINE_RUN}"
echo "  Run prefix:      ${C2K_RUN_PREFIX}"
echo "  Ks:              ${KS}"
echo "  last_n:          ${LAST_N}"
echo "  x_source:        ${X_SOURCE}"
echo "  include_c2:      ${INCLUDE_C2}"
echo "  output_dir:      ${OUTPUT_DIR}"
echo "=========================================================="

PYTHONPATH=./src:. "$PYTHON_BIN" scripts/eval/c2k_pareto_plot.py \
    "${ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Done."
echo "  Plot: ${OUTPUT_DIR}/c2k_pareto.png"
echo "  CSV:  ${OUTPUT_DIR}/c2k_pareto.csv"
echo "  Log:  ${LOG_FILE}"
