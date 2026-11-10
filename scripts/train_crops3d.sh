# Single-GPU training (GPU 0)
export PYTHONUNBUFFERED=1
echo "[$(date '+%H:%M:%S')] starting training (deps may take 2-5 min)..."
python -u tools/train_crops3d.py configs/default_crops3d.yaml \
    --gpu_ids 0 --num_gpus 1
