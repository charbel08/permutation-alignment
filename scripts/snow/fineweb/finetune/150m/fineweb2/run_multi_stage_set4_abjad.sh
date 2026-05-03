#!/bin/bash
set -euo pipefail

# Set 4 — Abjad / RTL scripts.
# Stages: C_2=Arabic, C_3=Hebrew, C_4=Persian.
# Reuses the standard 5% keys and the regular cumulative pretrain.

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint} \
RUN_SUFFIX=_set4_abjad \
LANGS="arb_Arab heb_Hebr fas_Arab" \
bash "$(dirname "$0")/run_multi_stage_cumulative.sh" "$@"
