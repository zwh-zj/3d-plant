#!/bin/bash
# Steps 0-3: split + text labels + preprocess clouds (full Crops3D)
set -e
cd "$(dirname "$0")/.."

CROPS_ROOT="Crops3D/Crops3D"

echo "========== data source: ${CROPS_ROOT} (full clouds) =========="
echo "note: large PLY preprocess is slow; training downsamples again via max_npoint in configs/default_crops3d.yaml."
echo ""

echo "========== Step 0: official train 30% -> val (crop x organ stratified); test unchanged =========="
python tools/split_train_val.py \
  --official_split_dir "${CROPS_ROOT}" \
  --crops_root "${CROPS_ROOT}" \
  --val_ratio 0.3 \
  --min_val_scenes_per_organ 2 \
  --seed 1999

echo ""
echo "========== Step 1: generate text-query JSONL =========="
python batch_all_crops.py --crops_root "${CROPS_ROOT}"

echo ""
echo "========== Step 2: JSONL -> crops3d JSON (train/val/test) =========="
python tools/convert_jsonl_to_crops3d_json.py

echo ""
echo "========== Step 3: PLY -> _refer.pth =========="
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" && conda activate 3d-gres 2>/dev/null || true
fi
python data/crops3d/prepare_crops3d.py \
  --crops_root "${CROPS_ROOT}" \
  --num_workers 4

echo ""
echo "Data prep done (full Crops3D)."
echo "  train: bash scripts/train_crops3d.sh"
echo "  test: bash scripts/test_crops3d.sh exps/default_crops3d/xxx/best.pth"
