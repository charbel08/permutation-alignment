#!/bin/bash
set -euo pipefail

source /work/.bashrc

export HF_HOME=${HF_HOME:-/work/scratch/hf}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/work/scratch/hf}

cd /work/permutation-alignment
mkdir -p outputs

# -----------------------------------------------------------------------------
# Deterministic inference sanity check with clean prompts per language.
# Runs one manual prompt each for EN, SPA, DEU, TUR through C1..C4 and writes
# all output to a single text file.
# -----------------------------------------------------------------------------

PYTHON_BIN=${PYTHON_BIN:-python3}
CHECKPOINT=${CHECKPOINT:-/work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_3langs_multi/stage3_spa_Latn_key3/final}
KEY_1=${KEY_1:-configs/keys/150m/both/key_7pct_1.json}
KEY_2=${KEY_2:-configs/keys/150m/both/key_7pct_2.json}
KEY_3=${KEY_3:-configs/keys/150m/both/key_7pct_3.json}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
ALLOW_TOKENIZER_DOWNLOAD=${ALLOW_TOKENIZER_DOWNLOAD:-0}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-25}
TEMPERATURE=${TEMPERATURE:-0}
OUTPUT_FILE=${OUTPUT_FILE:-outputs/clean_prompts_c1_c4.txt}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Missing Python executable: $PYTHON_BIN"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing CHECKPOINT: $CHECKPOINT"
    exit 1
fi

for key_file in "$KEY_1" "$KEY_2" "$KEY_3"; do
    if [ ! -f "$key_file" ]; then
        echo "Missing key file: $key_file"
        exit 1
    fi
done

mkdir -p "$(dirname "$OUTPUT_FILE")"
: > "$OUTPUT_FILE"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --key_paths "$KEY_1" "$KEY_2" "$KEY_3"
    --temperature "$TEMPERATURE"
    --max_new_tokens "$MAX_NEW_TOKENS"
)

if [ -n "$TOKENIZER_PATH" ]; then
    COMMON_ARGS+=(--tokenizer_path "$TOKENIZER_PATH")
fi

if [ "$ALLOW_TOKENIZER_DOWNLOAD" = "1" ]; then
    COMMON_ARGS+=(--allow_tokenizer_download)
fi

run_prompt() {
    local lang_tag="$1"
    local prompt_text="$2"

    {
        echo
        echo "==================== ${lang_tag} ===================="
        echo "PROMPT: ${prompt_text}"
        echo
    } >> "$OUTPUT_FILE"

    PYTHONPATH=./src "$PYTHON_BIN" -m tiered.train.inference \
        "${COMMON_ARGS[@]}" \
        --prompt "$prompt_text" >> "$OUTPUT_FILE"
}

echo "Running deterministic clean-prompt inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Output file: $OUTPUT_FILE"

run_prompt "EN" "In recent years, vaccine development has combined laboratory experiments with large clinical studies, and researchers have focused on"
run_prompt "SPA" "En los ultimos anos, el desarrollo de vacunas ha combinado experimentos de laboratorio con estudios clinicos amplios, y los investigadores se han centrado en"
run_prompt "DEU" "In den letzten Jahren hat die Impfstoffentwicklung Laborversuche mit grossen klinischen Studien kombiniert, und die Forscher haben sich auf"
run_prompt "TUR" "Son yillarda asi gelistirme sureci laboratuvar deneylerini genis capli klinik calismalarla birlestirdi ve arastirmacilar"

echo "Done. Results saved to: $OUTPUT_FILE"
