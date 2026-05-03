#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Per-checkpoint LLM-as-judge sweep for the MIXED-variant 530M instruction
# tune (output of run_instruction_tune_mix.sh, no KL ref).
#
# Mirrors run_llm_judge_instruction_tune_checkpoints.sh but defaults to the
# mix training output directory and shells out to run_llm_judge_c1_vs_c2_mix.sh
# so the inner generation/judging path matches the standalone mix evaluator.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}

# Mixed loss weights — must match the run_instruction_tune_mix.sh values
# whose checkpoints you want to evaluate.
W_PRIV=${W_PRIV:-0.8}
W_PUB_C2=${W_PUB_C2:-0.1}
W_PUB_C1=${W_PUB_C1:-0.1}
PRIV_TAG=${W_PRIV//./p}
PUB2_TAG=${W_PUB_C2//./p}
PUB1_TAG=${W_PUB_C1//./p}
WEIGHT_TAG=priv${PRIV_TAG}_pub2${PUB2_TAG}_pub1${PUB1_TAG}

BASE_CHECKPOINT=${BASE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_530m_${KEY_SIZE}pct/final-checkpoint}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/work/scratch/checkpoints/fineweb/instruction_tune_530m_alpaca_mix_key${KEY_SIZE}pct_${WEIGHT_TAG}}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/530m/both/key_${KEY_SIZE}pct.json}
EVAL_ROOT=${EVAL_ROOT:-/work/scratch/checkpoints/fineweb/evals/llm_judge_instruction_tune_530m_alpaca_mix_key${KEY_SIZE}pct_${WEIGHT_TAG}}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_INSTANCES=${MAX_INSTANCES:-}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-1.0}
DO_SAMPLE=${DO_SAMPLE:-0}

JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-4}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-1024}
DIFFICULTY_MODE=${DIFFICULTY_MODE:-auto}
DIFFICULTY_FIELD=${DIFFICULTY_FIELD:-difficulty}

INCLUDE_BASE_CHECKPOINT=${INCLUDE_BASE_CHECKPOINT:-1}
INCLUDE_FINAL=${INCLUDE_FINAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-main-instr-tune-mix}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_LOG_RESULTS=${WANDB_LOG_RESULTS:-1}

if [ ! -d "$TRAIN_OUTPUT_DIR" ]; then
    echo "Missing TRAIN_OUTPUT_DIR: $TRAIN_OUTPUT_DIR"
    exit 1
fi
if [ ! -f "$KEY_PATH" ]; then
    echo "Missing KEY_PATH: $KEY_PATH"
    exit 1
fi

declare -a CHECKPOINT_NAMES=()
declare -A CHECKPOINT_PATHS
declare -A CHECKPOINT_STEPS

if [ "$INCLUDE_BASE_CHECKPOINT" = "1" ]; then
    if [ ! -d "$BASE_CHECKPOINT" ]; then
        echo "Missing BASE_CHECKPOINT: $BASE_CHECKPOINT"
        exit 1
    fi
    CHECKPOINT_NAMES+=("base")
    CHECKPOINT_PATHS["base"]="$BASE_CHECKPOINT"
    CHECKPOINT_STEPS["base"]=0
fi

mapfile -t TRAIN_CHECKPOINT_NAMES < <(
    find "$TRAIN_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
)
for checkpoint_name in "${TRAIN_CHECKPOINT_NAMES[@]}"; do
    CHECKPOINT_NAMES+=("$checkpoint_name")
    CHECKPOINT_PATHS["$checkpoint_name"]="$TRAIN_OUTPUT_DIR/$checkpoint_name"
done

if [ "$INCLUDE_FINAL" = "1" ] && [ -d "$TRAIN_OUTPUT_DIR/final" ]; then
    CHECKPOINT_NAMES+=("final")
    CHECKPOINT_PATHS["final"]="$TRAIN_OUTPUT_DIR/final"
fi

if [ "${#CHECKPOINT_NAMES[@]}" -eq 0 ]; then
    echo "No checkpoints found to evaluate"
    exit 1
fi

mkdir -p "$EVAL_ROOT"

DEFAULT_WANDB_GROUP=$(basename "$TRAIN_OUTPUT_DIR")
WANDB_GROUP=${WANDB_GROUP:-$DEFAULT_WANDB_GROUP}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${DEFAULT_WANDB_GROUP}_llm_judge}
WANDB_RUN_ID_FILE=${WANDB_RUN_ID_FILE:-$EVAL_ROOT/wandb_eval_run_id.txt}
WANDB_RUN_ID=""
if [ "$WANDB_LOG_RESULTS" = "1" ]; then
    if [ -f "$WANDB_RUN_ID_FILE" ]; then
        WANDB_RUN_ID=$(tr -d '\n' < "$WANDB_RUN_ID_FILE")
    else
        WANDB_RUN_ID=$(python3 - <<'PY'
import uuid
print(uuid.uuid4().hex[:8])
PY
)
        printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
    fi
fi

echo "=========================================================="
echo "LLM-as-judge sweep over instruction-tuning checkpoints (MIX)"
echo "  Train output:  ${TRAIN_OUTPUT_DIR}"
echo "  Eval root:     ${EVAL_ROOT}"
echo "  Key path:      ${KEY_PATH}"
echo "  Loss weights:  w_priv=${W_PRIV} w_pub_c2=${W_PUB_C2} w_pub_c1=${W_PUB_C1}"
echo "  GPUs:          ${NGPUS}"
echo "  Judge model:   ${JUDGE_MODEL}"
if [ "$WANDB_LOG_RESULTS" = "1" ]; then
    echo "  W&B project:   ${WANDB_PROJECT}"
    echo "  W&B run:       ${WANDB_RUN_NAME} (${WANDB_RUN_ID})"
fi
echo "  Checkpoints:   ${#CHECKPOINT_NAMES[@]}"
for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    echo "    - ${checkpoint_name}"
done
echo "=========================================================="

for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    checkpoint_path="${CHECKPOINT_PATHS[$checkpoint_name]}"
    eval_output_dir="${EVAL_ROOT}/${checkpoint_name}"

    echo ""
    echo ">>> Evaluating ${checkpoint_name}"

    KEY_SIZE="$KEY_SIZE" \
    W_PRIV="$W_PRIV" \
    W_PUB_C2="$W_PUB_C2" \
    W_PUB_C1="$W_PUB_C1" \
    CHECKPOINT="$checkpoint_path" \
    KEY_PATH="$KEY_PATH" \
    OUTPUT_DIR="$eval_output_dir" \
    NGPUS="$NGPUS" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    MAX_INSTANCES="$MAX_INSTANCES" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    DO_SAMPLE="$DO_SAMPLE" \
    JUDGE_MODEL="$JUDGE_MODEL" \
    JUDGE_BATCH_SIZE="$JUDGE_BATCH_SIZE" \
    JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
    DIFFICULTY_MODE="$DIFFICULTY_MODE" \
    DIFFICULTY_FIELD="$DIFFICULTY_FIELD" \
    bash scripts/snow/fineweb/finetune/530m/alpaca/run_llm_judge_c1_vs_c2_mix.sh

    if [ "$WANDB_LOG_RESULTS" = "1" ]; then
        WANDB_ARGS=(--project "$WANDB_PROJECT")
        if [ -n "$WANDB_ENTITY" ]; then
            WANDB_ARGS+=(--entity "$WANDB_ENTITY")
        fi
        STEP_ARGS=()
        if [ -n "${CHECKPOINT_STEPS[$checkpoint_name]+x}" ]; then
            STEP_ARGS+=(--step_override "${CHECKPOINT_STEPS[$checkpoint_name]}")
        fi

        python3 scripts/eval/log_llm_judge_checkpoint_to_wandb.py \
            --checkpoint_dir "$checkpoint_path" \
            --results_path "${eval_output_dir}/judge_results.json" \
            --run_id "$WANDB_RUN_ID" \
            --run_name "$WANDB_RUN_NAME" \
            --group "$WANDB_GROUP" \
            --checkpoint_name "$checkpoint_name" \
            "${STEP_ARGS[@]}" \
            "${WANDB_ARGS[@]}"
    fi
done
