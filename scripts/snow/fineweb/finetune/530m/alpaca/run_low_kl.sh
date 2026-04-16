#!/bin/bash
set -euo pipefail

# Wrapper for Alpaca finetuning with a lower KL coefficient.
# This wrapper hard-sets KL_LAMBDA=0.03.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export KL_LAMBDA="0.03"

exec bash "${SCRIPT_DIR}/run.sh"
