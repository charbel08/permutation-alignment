#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=/work/scratch/hf
export TRANSFORMERS_CACHE=/work/scratch/hf

cd /work/permutation-alignment
mkdir -p logs

# ---------------------------------------------------------------------------
# Extraction attack on synthetic bios (150M).
#
# For each data fraction, finetune:
#   1. Tiered-finetuned model (attack C1 without key)
#   2. Baseline pretrained model
# Both early-stop when eval loss reaches C2 target.
# ---------------------------------------------------------------------------

KEY_SIZE=${KEY_SIZE:-5}
KL_TAG=${KL_TAG:-0p1}

TIERED_CHECKPOINT=${TIERED_CHECKPOINT:-/work/scratch/checkpoints/fineweb/private_finetune_150m_synbios_key${KEY_SIZE}pct_kl${KL_TAG}/final}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-/work/scratch/checkpoints/fineweb/baseline_pretrain_150m/final-checkpoint}
KEY_PATH=${KEY_PATH:-/work/permutation-alignment/configs/keys/150m/both/key_${KEY_SIZE}pct.json}

PRIVATE_DATA=${PRIVATE_DATA:-/work/scratch/data/datasets/synthetic_bios/tokenized}

NGPUS=${NGPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
MAX_STEPS=${MAX_STEPS:-50000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_STEPS=${EVAL_STEPS:-50}
PATIENCE=${PATIENCE:-5000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-extraction-attack}

FRACTIONS=${FRACTIONS:-"0.01 0.02 0.05 0.10 0.20 0.30 0.40 0.50 0.75 1.00"}

RUN_TIERED=${RUN_TIERED:-1}
RUN_BASELINE=${RUN_BASELINE:-1}

OUTPUT_BASE=${OUTPUT_BASE:-/work/scratch/checkpoints/fineweb/extraction_attack_150m_synbios_key${KEY_SIZE}pct}

echo "=========================================================="
echo "Extraction Attack — Synthetic Bios (150M)"
echo "  Key size:            ${KEY_SIZE}%"
echo "  Tiered checkpoint:   ${TIERED_CHECKPOINT}"
echo "  Baseline checkpoint: ${BASELINE_CHECKPOINT}"
echo "  Key path:            ${KEY_PATH}"
echo "  Private data:        ${PRIVATE_DATA}"
echo "  Fractions:           ${FRACTIONS}"
echo "  GPUs:                ${NGPUS}"
echo "  Max steps/run:       ${MAX_STEPS}"
echo "  Patience:            ${PATIENCE}"
echo "=========================================================="

# ── Measure C2 target loss once from the tiered model ──
echo ""
echo ">>> Measuring C2 target loss from tiered model..."
C2_TARGET=$(python - "$TIERED_CHECKPOINT" "$KEY_PATH" "$PRIVATE_DATA" "$BATCH_SIZE" "$EVAL_STEPS" <<'PY'
import sys, torch, math
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key
from tiered.permutation.permute import apply_permutation, unapply_permutation, build_swap_plan

checkpoint, key_path, data_path = sys.argv[1], sys.argv[2], sys.argv[3]
batch_size, eval_steps = int(sys.argv[4]), int(sys.argv[5])

device = torch.device("cuda")
model = GPTNeoForCausalLMTiered.from_pretrained(checkpoint).to(device)
key = load_key(key_path)
plan = build_swap_plan(model, key, device)

ds = load_from_disk(data_path)
val = ds["test"] if "test" in ds else ds.select(range(min(1000, len(ds))))
cols = [c for c in val.column_names if c not in ("input_ids", "attention_mask")]
if cols:
    val = val.remove_columns(cols)

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                    collate_fn=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
                    drop_last=True)

model.eval()
total, n = 0.0, 0
it = iter(loader)
with torch.no_grad():
    for _ in range(eval_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ids = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        apply_permutation(model, key, plan=plan)
        total += model(ids, labels=labs).loss.item()
        unapply_permutation(model, key, plan=plan)
        n += 1

print(f"{total / n:.6f}")
PY
)

echo "C2 target loss: ${C2_TARGET}"

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
            --target_loss "$C2_TARGET" \
            --private_data "$PRIVATE_DATA" \
            --data_fraction "$FRAC" \
            --output_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --min_lr "$MIN_LR" \
            --max_steps "$MAX_STEPS" \
            --warmup_steps "$WARMUP_STEPS" \
            --eval_interval "$EVAL_INTERVAL" \
            --eval_steps "$EVAL_STEPS" \
            --patience "$PATIENCE" \
            --num_workers "$NUM_WORKERS" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "$RUN_NAME" \
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
            --target_loss "$C2_TARGET" \
            --private_data "$PRIVATE_DATA" \
            --data_fraction "$FRAC" \
            --output_dir "$OUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --min_lr "$MIN_LR" \
            --max_steps "$MAX_STEPS" \
            --warmup_steps "$WARMUP_STEPS" \
            --eval_interval "$EVAL_INTERVAL" \
            --eval_steps "$EVAL_STEPS" \
            --patience "$PATIENCE" \
            --num_workers "$NUM_WORKERS" \
            --wandb_project "$WANDB_PROJECT" \
            --run_name "$RUN_NAME" \
            2>&1 | tee "$LOG_FILE"
    fi
done

echo ""
echo "=========================================================="
echo "All extraction attack runs complete."
echo "=========================================================="
