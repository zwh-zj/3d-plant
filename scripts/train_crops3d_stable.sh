#!/usr/bin/env bash
# Crops3D stable training: single process, no debug; config has preload / pin_memory=false / save_iter
set -euo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "train_crops3d.py" >/dev/null 2>&1; then
  echo "[error] train_crops3d is already running; run: pkill -9 -f train_crops3d.py"
  echo "       If it persists, run in Windows PowerShell: wsl --shutdown"
  exit 1
fi

echo "starting training (100 epoch, GPU 0, with val)..."
exec python -u tools/train_crops3d.py configs/default_crops3d.yaml --gpu_ids 0 --num_gpus 1 "$@"
