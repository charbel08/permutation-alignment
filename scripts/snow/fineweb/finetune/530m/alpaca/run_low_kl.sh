#!/bin/bash
set -euo pipefail

# Wrapper for Alpaca finetuning with a lower KL coefficient by default.
# You can still override any variable at launch time, e.g.:
#   KL_LAMBDA=0.01 KEY_SIZE=5 bash .../run_low_kl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KL_LAMBDA="${KL_LAMBDA:-0.03}"

exec bash "${SCRIPT_DIR}/run.sh"
