#!/bin/bash
set -euo pipefail

# Multi-stage mixed-data finetune: priv α = 0.9, public bundle = 0.05,
# each-anchor = 0.05 (exact at stage 1; balance shifts at other stages
# per the share-factor renormalization).
#
# Wraps run_multi_stage_cumulative_mix.sh with the corresponding lambdas.
# All other env-var overrides (KEY_SIZE, NGPUS, MAX_STEPS, etc.) pass through.

PUB_LAMBDA=0.1 \
ANCHOR_LAMBDA=0.1 \
EXPERIMENT_TAG=mix_priv0p9 \
    bash "$(dirname "$0")/run_multi_stage_cumulative_mix.sh" "$@"
