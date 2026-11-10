#!/usr/bin/env bash
# Module ablation trainer. Data/recipe match group A 20260720_182330.
# Output: exps/crops3d_full_baseline_fpfix_v15_module_ablation/YYYYMMDD_HHMMSS/
#
# Edit the four switches before running:
#   configs/crops3d_full_baseline_fpfix_v15_module_ablation.yaml
#     model.mdin.use_tsc / use_csa / use_crr / use_lca
#
# Single GPU:
#   bash scripts/train_crops3d_full_fpfix_v15_module_ablation.sh
#
# Choose GPU:
#   GPU_IDS=2 bash scripts/train_crops3d_full_fpfix_v15_module_ablation.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# This repo, plus the compiled pointgroup_ops extension under gres_model/lib.
export PYTHONPATH="$(pwd):$(pwd)/gres_model/lib${PYTHONPATH:+:$PYTHONPATH}"
# WSL GPU driver library. cuDNN dlopen fails without it on long runs.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /home/zwh/miniconda3/envs/3d-gres/bin/python ]]; then
    PYTHON=/home/zwh/miniconda3/envs/3d-gres/bin/python
  else
    PYTHON=/home/c_zhang/.conda/envs/3d-plant/bin/python
  fi
fi
ANN_JSON="data/crops3d/crops3d_train.json"
SAMPLE_PTH="data/crops3d/train/Cabbage_mvs_1005_01_refer.pth"
CFG="configs/crops3d_full_baseline_fpfix_v15_module_ablation.yaml"

if [[ ! -f "${ANN_JSON}" ]]; then
  echo "Missing ${ANN_JSON}; prepare full annotations first."
  exit 1
fi

if [[ ! -f "${SAMPLE_PTH}" ]]; then
  echo "Missing full pth files; filling preprocessing..."
  "$PYTHON" -u tools/prepare_missing_full_pth.py \
    --annotation_dir data/crops3d/all_crops_full \
    --num_workers 4
fi

# Print ablation switches to avoid the wrong combo.
echo "Ablation switches from ${CFG}:"
"$PYTHON" - << PY
import gorilla
cfg = gorilla.Config.fromfile("${CFG}")
m = cfg.model.mdin
print(f"  use_tsc={m.use_tsc}  use_csa={m.use_csa}  use_crr={m.use_crr}  use_lca={m.use_lca}")
PY

export PYTHONUNBUFFERED=1
NUM_GPUS="${NUM_GPUS:-1}"
# shellcheck disable=SC2206
GPU_IDS=(${GPU_IDS:-0})

echo "[$(date '+%H:%M:%S')] start module-ablation training (GPU: ${GPU_IDS[*]}, num_gpus=${NUM_GPUS})..."
"$PYTHON" -u tools/train_crops3d.py "${CFG}" \
  --gpu_ids "${GPU_IDS[@]}" --num_gpus "${NUM_GPUS}"
