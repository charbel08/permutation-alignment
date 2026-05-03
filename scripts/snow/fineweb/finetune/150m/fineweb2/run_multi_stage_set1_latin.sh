#!/bin/bash
set -euo pipefail

# Set 1 — Latin script, low-diacritic Romance + Slavic-Latin.
# Stages: C_2=Portuguese, C_3=Polish, C_4=Italian.
# Reuses the standard 5% keys (key_5pct_{1,2,3}.json) and the regular
# tiered cumulative pretrain checkpoint.

RUN_SUFFIX=_set1_latin \
LANGS="por_Latn pol_Latn ita_Latn" \
bash "$(dirname "$0")/run_multi_stage_cumulative.sh" "$@"
