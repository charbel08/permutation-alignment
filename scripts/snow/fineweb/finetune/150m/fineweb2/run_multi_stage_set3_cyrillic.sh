#!/bin/bash
set -euo pipefail

# Set 3 — Cyrillic / Greek alphabets.
# Stages: C_2=Russian, C_3=Ukrainian, C_4=Greek.
# Reuses the standard 5% keys and the regular cumulative pretrain.

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint} \
RUN_SUFFIX=_set3_cyrillic \
LANGS="rus_Cyrl ukr_Cyrl ell_Grek" \
bash "$(dirname "$0")/run_multi_stage_cumulative.sh" "$@"
