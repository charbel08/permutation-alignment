#!/bin/bash
set -euo pipefail

# Set 2 — Latin script, harder morphology (agglutinative / tonal).
# Stages: C_2=Vietnamese, C_3=Finnish, C_4=Hungarian.
# Reuses the standard 5% keys and the regular cumulative pretrain.

PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-/work/scratch/checkpoints/fineweb/tiered_pretrain_150m_5pct_multi_cumulative/final-checkpoint} \
RUN_SUFFIX=_set2_morpho \
LANGS="vie_Latn fin_Latn hun_Latn" \
bash "$(dirname "$0")/run_multi_stage_cumulative.sh" "$@"
