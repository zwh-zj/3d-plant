#!/usr/bin/env bash
# Final eval on the official test split (run after training).
# Usage:
#   bash scripts/test_crops3d.sh exps/crops3d_full_baseline_fpfix_v15/<timestamp>/best.pth
#   bash scripts/test_crops3d.sh <best.pth> structured
#   bash scripts/test_crops3d.sh <best.pth> both

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/gres_model/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CHECKPOINT=${1:?"Need checkpoint path, e.g. bash scripts/test_crops3d.sh exps/crops3d_full_baseline_fpfix_v15/<timestamp>/best.pth"}
TEST_MODE=${2:-canonical}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /home/zwh/miniconda3/envs/3d-gres/bin/python ]]; then
    PYTHON=/home/zwh/miniconda3/envs/3d-gres/bin/python
  else
    PYTHON=python
  fi
fi

NUM_GPUS="${NUM_GPUS:-1}"
# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS:-0})

run_test() {
  "$PYTHON" -u tools/test_crops3d.py configs/crops3d_full_baseline_fpfix_v15.yaml \
    --checkpoint "${CHECKPOINT}" \
    --test_mode "$1" \
    --gpu_ids "${GPU_IDS[@]}" --num_gpus "${NUM_GPUS}"
}

case "${TEST_MODE}" in
  canonical|structured)
    run_test "${TEST_MODE}"
    ;;
  both)
    run_test canonical
    run_test structured
    ;;
  *)
    echo "Unknown test mode: ${TEST_MODE}"
    echo "Choose: canonical | structured | both"
    exit 1
    ;;
esac
