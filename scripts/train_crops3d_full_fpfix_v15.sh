#!/usr/bin/env bash
# All crops, full point clouds, same recipe as maize fpfix_v15 (20260716_153329)
# Output: exps/crops3d_full_baseline_fpfix_v15/YYYYMMDD_HHMMSS/
#
# Single GPU (default GPU 0):
#   bash scripts/train_crops3d_full_fpfix_v15.sh
#
# Two GPUs:
#   NUM_GPUS=2 GPU_IDS="0 1" bash scripts/train_crops3d_full_fpfix_v15.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /home/zwh/miniconda3/envs/3d-gres/bin/python ]]; then
    PYTHON=/home/zwh/miniconda3/envs/3d-gres/bin/python
  else
    PYTHON=python
  fi
fi

ANN_JSON="data/crops3d/crops3d_train.json"
SAMPLE_PTH="data/crops3d/train/Cabbage_mvs_1005_01_refer.pth"

if [[ ! -f "${ANN_JSON}" ]]; then
  echo "Missing ${ANN_JSON}; prepare full annotations first."
  exit 1
fi

# Reject 10k scene_id mix-ups
if grep -q '_10k' "${ANN_JSON}"; then
  echo "error: ${ANN_JSON} contains _10k scene_id; not full-cloud annotations."
  exit 1
fi

if [[ ! -f "${SAMPLE_PTH}" ]]; then
  echo "Missing full pth files; filling preprocessing (skip existing)..."
  "$PYTHON" -u tools/prepare_missing_full_pth.py \
    --annotation_dir data/crops3d \
    --num_workers 4
fi

export PYTHONUNBUFFERED=1
NUM_GPUS="${NUM_GPUS:-1}"
# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS:-0})

echo "[$(date '+%H:%M:%S')] start full-crop fpfix_v15 training (GPU: ${GPU_IDS[*]}, num_gpus=${NUM_GPUS})..."
"$PYTHON" -u tools/train_crops3d.py configs/crops3d_full_baseline_fpfix_v15.yaml \
  --gpu_ids "${GPU_IDS[@]}" --num_gpus "${NUM_GPUS}"
