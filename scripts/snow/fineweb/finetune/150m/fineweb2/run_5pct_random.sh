#!/bin/bash
set -euo pipefail

# Private-finetune the 5% random-key 150M/180M tiered pretrain on Spanish FineWeb2.
# This wraps run.sh so the random-key checkpoint/key/output naming stay consistent.
#
# Expected default inputs:
#   /work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_random/final-checkpoint
#   /work/permutation-alignment/configs/keys/150m/both/key_5pct_random.json
#
# Output:
#   /work/scratch/checkpoints/fineweb/private_finetune_150m_fineweb2_spa_key5pct_random_kl0p1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KEY_SIZE="${KEY_SIZE:-5}" \
KEY_SUFFIX="${KEY_SUFFIX:-_random}" \
"${SCRIPT_DIR}/run.sh"
