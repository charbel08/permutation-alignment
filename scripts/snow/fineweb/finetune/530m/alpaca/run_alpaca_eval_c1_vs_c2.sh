#!/bin/bash
set -euo pipefail

source /work/.bashrc

cd /work/permutation-alignment
mkdir -p logs

KEY_SIZE="${KEY_SIZE:-5}"
KL_TAG="${KL_TAG:-0p03}"

JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/llm_judge_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}"
C1_OUTPUTS="${C1_OUTPUTS:-$JUDGE_OUTPUT_DIR/c1_outputs.json}"
C2_OUTPUTS="${C2_OUTPUTS:-$JUDGE_OUTPUT_DIR/c2_outputs.json}"

ALPACA_IO_DIR="${ALPACA_IO_DIR:-$JUDGE_OUTPUT_DIR/alpaca_eval_inputs}"
ALPACA_RESULTS_DIR="${ALPACA_RESULTS_DIR:-$JUDGE_OUTPUT_DIR/alpaca_eval_results}"

ANNOTATOR_MODE="${ANNOTATOR_MODE:-local}"  # local | openai_compat
if [ "$ANNOTATOR_MODE" = "openai_compat" ]; then
  ANNOTATOR_CONFIG="${ANNOTATOR_CONFIG:-/work/permutation-alignment/scripts/eval/alpaca_eval_configs/gpt_oss_120b_openai_compat.yaml}"
else
  ANNOTATOR_CONFIG="${ANNOTATOR_CONFIG:-/work/permutation-alignment/scripts/eval/alpaca_eval_configs/gpt_oss_120b_local.yaml}"
fi

if [ ! -f "$C1_OUTPUTS" ]; then
  echo "Missing C1 outputs: $C1_OUTPUTS"
  exit 1
fi
if [ ! -f "$C2_OUTPUTS" ]; then
  echo "Missing C2 outputs: $C2_OUTPUTS"
  exit 1
fi
if [ ! -f "$ANNOTATOR_CONFIG" ]; then
  echo "Missing annotator config: $ANNOTATOR_CONFIG"
  exit 1
fi

mkdir -p "$ALPACA_IO_DIR" "$ALPACA_RESULTS_DIR"

if ! command -v alpaca_eval >/dev/null 2>&1; then
  python3 -m pip install --user -U alpaca-eval
fi

python3 scripts/eval/export_llm_judge_outputs_to_alpaca_eval.py \
  --c1_outputs "$C1_OUTPUTS" \
  --c2_outputs "$C2_OUTPUTS" \
  --out_dir "$ALPACA_IO_DIR" \
  --model_name "tiered_c2_530m_key${KEY_SIZE}pct_kl${KL_TAG}" \
  --reference_name "tiered_c1_530m_key${KEY_SIZE}pct_kl${KL_TAG}"

MODEL_OUTPUTS="$ALPACA_IO_DIR/model_outputs.json"
REFERENCE_OUTPUTS="$ALPACA_IO_DIR/reference_outputs.json"

LOG_FILE="logs/alpaca_eval_530m_key${KEY_SIZE}pct_kl${KL_TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================================="
echo "AlpacaEval C2 vs C1 (Snow)"
echo "  Mode:            ${ANNOTATOR_MODE}"
echo "  Annotator cfg:   ${ANNOTATOR_CONFIG}"
echo "  Model outputs:   ${MODEL_OUTPUTS}"
echo "  Ref outputs:     ${REFERENCE_OUTPUTS}"
echo "  Results dir:     ${ALPACA_RESULTS_DIR}"
echo "=========================================================="

# Prefer explicit output_path; fall back for older/newer CLI variants.
set +e
alpaca_eval \
  --model_outputs "$MODEL_OUTPUTS" \
  --reference_outputs "$REFERENCE_OUTPUTS" \
  --annotators_config "$ANNOTATOR_CONFIG" \
  --output_path "$ALPACA_RESULTS_DIR" \
  2>&1 | tee "$LOG_FILE"
rc=$?
set -e

if [ $rc -ne 0 ]; then
  echo "Retrying alpaca_eval without --output_path..."
  alpaca_eval \
    --model_outputs "$MODEL_OUTPUTS" \
    --reference_outputs "$REFERENCE_OUTPUTS" \
    --annotators_config "$ANNOTATOR_CONFIG" \
    2>&1 | tee -a "$LOG_FILE"
fi

echo ""
echo "Done. Log: $LOG_FILE"
