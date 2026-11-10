#!/bin/bash
# Final eval on the official test split (run after training).
# Usage:
#   bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth
#   bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth canonical
#   bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth structured
#   bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth both

set -e
cd "$(dirname "$0")/.."

CHECKPOINT=${1:?"Need checkpoint path, e.g. bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth"}
TEST_MODE=${2:-canonical}

run_test() {
  local mode=$1
  python tools/test_crops3d.py configs/default_crops3d.yaml \
    --checkpoint "${CHECKPOINT}" \
    --test_mode "${mode}" \
    --gpu_ids 1 --num_gpus 1
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
