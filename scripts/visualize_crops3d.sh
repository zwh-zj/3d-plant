#!/bin/bash
# One-shot Crops3D segmentation visualization
# Usage:
#   bash scripts/visualize_crops3d.sh exps/default_crops3d/20260630_000109/best.pth Maize_1-111_10k soil

set -e
cd "$(dirname "$0")/.."

CHECKPOINT=${1:?"usage: bash scripts/visualize_crops3d.sh <best.pth> <scene_id> [organ]"}
SCENE_ID=${2:?"need scene_id, e.g. Maize_1-111_10k"}
ORGAN=${3:-soil}

if [ ! -f tools/render_ball.so ]; then
  echo "building render_ball.so ..."
  bash scripts/build_render_ball.sh
fi

PYTHON="${PYTHON:-/home/c_zhang/.conda/envs/3d-plant/bin/python}"
CONFIG="${CONFIG:-configs/maize_only_full.yaml}"

${PYTHON} tools/visualize_crops3d.py "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --scene_id "${SCENE_ID}" \
  --organ "${ORGAN}" \
  --split test \
  --export_dir "$(dirname "${CHECKPOINT}")/vis" \
  --no_show \
  --gpu_ids 0 --num_gpus 1
