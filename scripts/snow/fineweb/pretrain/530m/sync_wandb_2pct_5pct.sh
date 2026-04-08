#!/bin/bash
set -euo pipefail

source /work/.bashrc
cd /work/permutation-alignment

WANDB_ROOT=${WANDB_ROOT:-/work/permutation-alignment/wandb}
CKPT_ROOT=${CKPT_ROOT:-/work/scratch/checkpoints/fineweb}

sync_one() {
    local tag="$1"
    local ckpt_dir="$CKPT_ROOT/tiered_pretrain_530m_${tag}"

    local latest_ckpt
    latest_ckpt=$(ls -d "$ckpt_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [ -z "${latest_ckpt}" ]; then
        echo "[${tag}] No checkpoint found in: ${ckpt_dir}"
        return 1
    fi

    local state_path="${latest_ckpt}/training_state.pt"
    if [ ! -f "${state_path}" ]; then
        echo "[${tag}] Missing training_state.pt: ${state_path}"
        return 1
    fi

    local run_id
    run_id=$(python - <<PY
import torch
d = torch.load("${state_path}", map_location="cpu")
rid = d.get("wandb_run_id", "")
print(rid if rid is not None else "")
PY
)
    run_id=$(echo "${run_id}" | tr -d '[:space:]')
    if [ -z "${run_id}" ]; then
        echo "[${tag}] No wandb_run_id in: ${state_path}"
        return 1
    fi

    mapfile -t run_dirs < <(find "${WANDB_ROOT}" -maxdepth 1 -type d -name "run-*-${run_id}" | sort)
    if [ "${#run_dirs[@]}" -eq 0 ]; then
        echo "[${tag}] No local run dirs found for run_id=${run_id} under ${WANDB_ROOT}"
        echo "[${tag}] Check if this is the same machine/path where training ran."
        return 1
    fi

    echo "[${tag}] latest_ckpt: ${latest_ckpt}"
    echo "[${tag}] run_id: ${run_id}"
    echo "[${tag}] syncing ${#run_dirs[@]} local fragment(s):"
    printf '  - %s\n' "${run_dirs[@]}"

    wandb sync --include-online "${run_dirs[@]}"
}

echo "=== Syncing 530M pretrain run fragments (2pct, 5pct) ==="
sync_one "2pct" || true
echo
sync_one "5pct" || true
echo
echo "Done."
