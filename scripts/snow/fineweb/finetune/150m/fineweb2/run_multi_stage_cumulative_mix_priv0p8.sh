#!/bin/bash
set -euo pipefail

# Multi-stage mixed-data finetune.
# Per-term loss weights at every stage:
#   priv = 0.8  (single CE term)
#   public bundle = 0.1 (split equally across N+1 configs)
#   anchor bundle = 0.1 (split equally across t prior tiers; absent at stage 0)
# No renormalization — passed values ARE the weights.

W_PRIV=0.8 \
W_PUB=0.1 \
W_ANCHOR=0.1 \
EXPERIMENT_TAG=mix_priv0p8 \
    bash "$(dirname "$0")/run_multi_stage_cumulative_mix.sh" "$@"
