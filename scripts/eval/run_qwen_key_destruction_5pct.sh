#!/usr/bin/env bash
set -euo pipefail

# Run Qwen MMLU key-destruction ablation with key sizes in 5% increments.
#
# Example:
#   module load anaconda/3 cuda/12.6.0/cudnn openmpi
#   conda activate ta
#   ./scripts/eval/run_qwen_key_destruction_5pct.sh \
#     --model-id Qwen/Qwen3-8B \
#     --min-pct 0.05 \
#     --max-pct 1.00 \
#     --step-pct 0.05
#
# Output files are written to:
#   outputs/<run_name>.json
#   outputs/<run_name>.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="Qwen/Qwen3-8B"
TOKENIZER_ID=""
MIN_PCT="0.05"
MAX_PCT="1.00"
STEP_PCT="0.05"
ATTN_RATIO="0.25"
SEED="42"
SHOTS="5"
MAX_EXAMPLES_PER_SUBJECT="20"
DEVICE="cuda"
DTYPE="bfloat16"
DATASET_NAME="cais/mmlu"
TRUST_REMOTE_CODE="0"
WANDB="0"
WANDB_PROJECT="tiered-alignment-ablation"
WANDB_RUN_NAME=""
WANDB_ENTITY=""
RUN_NAME="qwen_key_destruction_5pct_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs"
SUBJECTS=()

usage() {
  cat <<'EOF'
Usage: run_qwen_key_destruction_5pct.sh [options]

Options:
  --model-id <id>                  HF model id (default: Qwen/Qwen3-8B)
  --tokenizer-id <id>              HF tokenizer id (default: model id)
  --min-pct <float>                Minimum key percent (default: 0.05)
  --max-pct <float>                Maximum key percent (default: 1.00)
  --step-pct <float>               Increment size (default: 0.05)
  --attn-ratio <float>             Attention ratio in key gen (default: 0.25)
  --seed <int>                     Base seed (default: 42)
  --shots <int>                    MMLU few-shot k (default: 5)
  --max-examples-per-subject <n>   Per-subject eval cap; -1 = full (default: 20)
  --device <cpu|cuda>              Device (default: cuda)
  --dtype <float16|bfloat16|float32>
                                    Dtype (default: bfloat16)
  --dataset-name <name>            HF dataset name (default: cais/mmlu)
  --subject <name>                 Restrict to subject (repeatable)
  --wandb                          Enable W&B logging
  --wandb-project <name>           W&B project (default: tiered-alignment-ablation)
  --wandb-run-name <name>          W&B run name (default: run-name)
  --wandb-entity <name>            W&B entity/team (optional)
  --trust-remote-code              Pass through to HF loader
  --run-name <name>                Output base name
  --output-dir <path>              Output dir (default: outputs)
  -h, --help                       Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --tokenizer-id) TOKENIZER_ID="$2"; shift 2 ;;
    --min-pct) MIN_PCT="$2"; shift 2 ;;
    --max-pct) MAX_PCT="$2"; shift 2 ;;
    --step-pct) STEP_PCT="$2"; shift 2 ;;
    --attn-ratio) ATTN_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --max-examples-per-subject) MAX_EXAMPLES_PER_SUBJECT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --subject) SUBJECTS+=("$2"); shift 2 ;;
    --wandb) WANDB="1"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --trust-remote-code) TRUST_REMOTE_CODE="1"; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

readarray -t KEY_PCTS < <(
  python - "${MIN_PCT}" "${MAX_PCT}" "${STEP_PCT}" <<'PY'
import sys
mn = float(sys.argv[1])
mx = float(sys.argv[2])
st = float(sys.argv[3])
if mn <= 0 or mx <= 0 or st <= 0:
    raise SystemExit("min/max/step must be > 0")
if mx < mn:
    raise SystemExit("max-pct must be >= min-pct")

vals = []
cur = mn
eps = 1e-12
while cur <= mx + eps:
    vals.append(f"{cur:.6f}".rstrip("0").rstrip("."))
    cur += st
print("\n".join(vals))
PY
)

if [[ "${#KEY_PCTS[@]}" -eq 0 ]]; then
  echo "No key percentages generated. Check min/max/step." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
OUT_JSON="${OUTPUT_DIR}/${RUN_NAME}.json"
OUT_CSV="${OUTPUT_DIR}/${RUN_NAME}.csv"

CMD=(
  python scripts/eval/mmlu_qwen_key_ablation.py
  --model_id "${MODEL_ID}"
  --key_pcts "${KEY_PCTS[@]}"
  --attn_ratio "${ATTN_RATIO}"
  --seed "${SEED}"
  --shots "${SHOTS}"
  --max_examples_per_subject "${MAX_EXAMPLES_PER_SUBJECT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --dataset_name "${DATASET_NAME}"
  --output_json "${OUT_JSON}"
  --output_csv "${OUT_CSV}"
)

if [[ -n "${TOKENIZER_ID}" ]]; then
  CMD+=(--tokenizer_id "${TOKENIZER_ID}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi
if [[ "${#SUBJECTS[@]}" -gt 0 ]]; then
  CMD+=(--subjects "${SUBJECTS[@]}")
fi
if [[ "${WANDB}" == "1" ]]; then
  CMD+=(--wandb --wandb_project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    CMD+=(--wandb_run_name "${WANDB_RUN_NAME}")
  else
    CMD+=(--wandb_run_name "${RUN_NAME}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

echo "Running Qwen key destruction ablation"
echo "  model: ${MODEL_ID}"
echo "  key_pcts: ${KEY_PCTS[*]}"
echo "  output: ${OUT_JSON}"
echo "  output: ${OUT_CSV}"
if [[ "${WANDB}" == "1" ]]; then
  echo "  wandb: project=${WANDB_PROJECT} run=${WANDB_RUN_NAME:-$RUN_NAME}"
fi

PYTHONPATH=./src "${CMD[@]}"
