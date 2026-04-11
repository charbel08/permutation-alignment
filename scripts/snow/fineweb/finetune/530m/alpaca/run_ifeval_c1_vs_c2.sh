#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# IFEval C1 vs C2 for 530M instruct-tuned tiered model.
#
# C1 (public tier) is used as baseline.
# All scoring is deterministic (no LLM judge).
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}/final}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}

OUTPUT_DIR=${OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/evals/ifeval_530m_alpaca_key${KEY_SIZE}pct_kl${KL_TAG}}
NGPUS=${NGPUS:-8}

MAX_INSTANCES=${MAX_INSTANCES:-}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

# Clone IFEval if not present
IFEVAL_DIR=/work/permutation-alignment/third_party/instruction_following_eval
if [ ! -d "$IFEVAL_DIR" ]; then
    echo "Cloning instruction_following_eval..."
    mkdir -p /work/permutation-alignment/third_party
    git clone --depth 1 https://github.com/google-research/google-research.git /tmp/google-research
    cp -r /tmp/google-research/instruction_following_eval "$IFEVAL_DIR"
    rm -rf /tmp/google-research
fi
uv pip install -q langdetect immutabledict

mkdir -p "$OUTPUT_DIR"

echo "=========================================================="
echo "IFEval C1 vs C2 (530M Alpaca)"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Key path:     ${KEY_PATH}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GPUs:         ${NGPUS}"
echo "  Batch size:   ${BATCH_SIZE}"
echo "  Max tokens:   ${MAX_NEW_TOKENS}"
echo "=========================================================="

EXTRA_ARGS=()
if [ -n "$MAX_INSTANCES" ]; then
  EXTRA_ARGS+=(--max_instances "$MAX_INSTANCES")
fi

LOG_FILE="logs/ifeval_c1_vs_c2_530m_key${KEY_SIZE}pct_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=./src:./third_party:. torchrun --standalone --nproc_per_node="$NGPUS" \
  scripts/eval/ifeval_c1_c2.py \
  --checkpoint "$CHECKPOINT" \
  --key_path "$KEY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results: ${OUTPUT_DIR}/ifeval_results.json"
echo "Log file:      ${LOG_FILE}"
