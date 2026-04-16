#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf
export TOKENIZERS_PARALLELISM=false

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Extraction attack on synthetic bios (150M).
#
# For each data fraction, finetune:
#   1. Tiered-finetuned model (attack C1 without key)
#   2. Baseline pretrained model
# Train for a fixed number of steps (no early stopping) to match the
# token budget used in private finetuning.
# data_fraction is applied to the train split only.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

TIERED_CHECKPOINT=${TIERED_CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}/final}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
TIERED_PRETRAIN_CHECKPOINT=${TIERED_PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_${KEY_SIZE}pct/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/synthetic_bios/tokenized}
BIO_METADATA=${BIO_METADATA:-/work/scratch/data/datasets/synthetic_bios/bios_metadata.json}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-3e-5}
MIN_LR=${MIN_LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
MAX_STEPS=${MAX_STEPS:-4050}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-extraction-attack}
SUBSET_SEED=${SUBSET_SEED:-42}

FRACTIONS=${FRACTIONS:-"0.50 1.00"}

RUN_TIERED=${RUN_TIERED:-1}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_TIERED_PRETRAIN=${RUN_TIERED_PRETRAIN:-1}

OUTPUT_BASE=${OUTPUT_BASE:-/work/scratch/checkpoints/fineweb/extraction_attack_150m_synbios_key${KEY_SIZE}pct}

echo "=========================================================="
echo "Extraction Attack — Synthetic Bios (150M)"
echo "  Key size:            ${KEY_SIZE}%"
echo "  Tiered checkpoint:   ${TIERED_CHECKPOINT}"
echo "  Baseline checkpoint: ${BASELINE_CHECKPOINT}"
echo "  Tiered pretrain ckpt:${TIERED_PRETRAIN_CHECKPOINT}"
echo "  Key path:            ${KEY_PATH}"
echo "  Private data:        ${PRIVATE_DATA}"
echo "  Fractions:           ${FRACTIONS}"
echo "  GPUs:                ${NGPUS}"
echo "  Max steps:           ${MAX_STEPS} (fixed-step, no early stop)"
echo "  Subset seed:         ${SUBSET_SEED}"
echo "  Bio metadata:        ${BIO_METADATA}"
echo "=========================================================="
echo ""
echo "No early stopping: each run trains for MAX_STEPS to match token budget."
echo "train_people/test_people memorization is still logged for reporting."
echo ""

for FRAC in $FRACTIONS; do
    FRAC_TAG=${FRAC//./p}

    # ── Tiered attack ──
    if [ "$RUN_TIERED" = "1" ]; then
        RUN_NAME="attack_tiered_synbios_key${KEY_SIZE}pct_frac${FRAC_TAG}"
        OUT_DIR="${OUTPUT_BASE}/tiered/frac_${FRAC_TAG}"
        LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo ">>> Tiered attack: fraction=${FRAC}"

        torchrun --standalone --nproc_per_node="$NGPUS" \
            -m tiered.train.finetune.extraction_attack \
            --model_checkpoint "$TIERED_CHECKPOINT" \
            --tiered_checkpoint "$TIERED_CHECKPOINT" \
            --key_path "$KEY_PATH" \
            --private_data "$PRIVATE_DATA" \
            --data_fraction "$FRAC" \
            --subset_seed "$SUBSET_SEED" \
            --output_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --min_lr "$MIN_LR" \
            --warmup_steps "$WARMUP_STEPS" \
            --max_steps "$MAX_STEPS" \
            --eval_interval "$EVAL_INTERVAL" \
            --num_workers "$NUM_WORKERS" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "$RUN_NAME" \
            --bio_metadata "$BIO_METADATA" \
            2>&1 | tee "$LOG_FILE"
    fi

    # ── Baseline attack ──
    if [ "$RUN_BASELINE" = "1" ]; then
        RUN_NAME="attack_baseline_synbios_key${KEY_SIZE}pct_frac${FRAC_TAG}"
        OUT_DIR="${OUTPUT_BASE}/baseline/frac_${FRAC_TAG}"
        LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo ">>> Baseline attack: fraction=${FRAC}"

        torchrun --standalone --nproc_per_node="$NGPUS" \
            -m tiered.train.finetune.extraction_attack \
            --model_checkpoint "$BASELINE_CHECKPOINT" \
            --tiered_checkpoint "$TIERED_CHECKPOINT" \
            --key_path "$KEY_PATH" \
            --private_data "$PRIVATE_DATA" \
            --data_fraction "$FRAC" \
            --subset_seed "$SUBSET_SEED" \
            --output_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --min_lr "$MIN_LR" \
            --warmup_steps "$WARMUP_STEPS" \
            --max_steps "$MAX_STEPS" \
            --eval_interval "$EVAL_INTERVAL" \
            --num_workers "$NUM_WORKERS" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "$RUN_NAME" \
            --bio_metadata "$BIO_METADATA" \
            2>&1 | tee "$LOG_FILE"
    fi

    # ── Tiered-pretrain attack (not synthetic-data finetuned) ──
    if [ "$RUN_TIERED_PRETRAIN" = "1" ]; then
        RUN_NAME="attack_tiered_pretrain_synbios_key${KEY_SIZE}pct_frac${FRAC_TAG}"
        OUT_DIR="${OUTPUT_BASE}/tiered_pretrain/frac_${FRAC_TAG}"
        LOG_FILE="logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo ">>> Tiered-pretrain attack: fraction=${FRAC}"

        torchrun --standalone --nproc_per_node="$NGPUS" \
            -m tiered.train.finetune.extraction_attack \
            --model_checkpoint "$TIERED_PRETRAIN_CHECKPOINT" \
            --tiered_checkpoint "$TIERED_CHECKPOINT" \
            --key_path "$KEY_PATH" \
            --private_data "$PRIVATE_DATA" \
            --data_fraction "$FRAC" \
            --subset_seed "$SUBSET_SEED" \
            --output_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --min_lr "$MIN_LR" \
            --warmup_steps "$WARMUP_STEPS" \
            --max_steps "$MAX_STEPS" \
            --eval_interval "$EVAL_INTERVAL" \
            --num_workers "$NUM_WORKERS" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "$RUN_NAME" \
            --bio_metadata "$BIO_METADATA" \
            2>&1 | tee "$LOG_FILE"
    fi
done

echo ""
echo "=========================================================="
echo "All extraction attack runs complete."
echo "=========================================================="
