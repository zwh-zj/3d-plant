#!/usr/bin/env bash
# Resume from last_iter.pth (work_dir must match the crashed run)
set -euo pipefail
cd "$(dirname "$0")/.."

WORK_DIR="${1:?usage: bash scripts/resume_crops3d_stable.sh exps/default_crops3d/YYYYMMDD_HHMMSS}"
CKPT="${WORK_DIR}/last_iter.pth"

if [[ ! -f "$CKPT" ]]; then
  echo "[error] checkpoint not found: $CKPT"
  exit 1
fi

if pgrep -f "train_crops3d.py" >/dev/null 2>&1; then
  echo "[error] train_crops3d is already running"
  exit 1
fi

echo "resume: $CKPT"
exec python -u tools/train_crops3d.py configs/default_crops3d.yaml \
  --gpu_ids 0 --num_gpus 1 \
  --work_dir "$WORK_DIR" \
  --resume "$CKPT"
