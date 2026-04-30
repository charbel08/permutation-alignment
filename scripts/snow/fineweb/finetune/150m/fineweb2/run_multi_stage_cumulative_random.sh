#!/bin/bash
set -euo pipefail

# Multi-stage cumulative finetune on the RANDOM-key variant.
#
# Mirrors run_multi_stage_cumulative.sh but starts from the random
# cumulative pretrain (tiered_pretrain_150m_5pct_multi_cumulative_random)
# and uses the matching random keys (key_5pct_random_{1,2,3}).
#
# Output dir:
#   /work/scratch/checkpoints/fineweb/finetune_150m_fineweb2_multi_stage_random_perconfig_key5pct_kl0p1
#
# Pass-through env vars (anything you'd set on the base script also works
# here: KL_LAMBDA, BATCH_SIZE, MAX_STEPS, etc.).

RUN_SUFFIX=_random \
KEY_SUFFIX=_random \
bash "$(dirname "$0")/run_multi_stage_cumulative.sh" "$@"
