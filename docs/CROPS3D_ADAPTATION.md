# Crops3D Text-Query Organ Segmentation — MDIN Adaptation

> Task: text-query plant organ segmentation  
> Base framework: MDIN (3D-GRES)  
> Dataset: Crops3D
export CUDA_LAUNCH_BLOCKING=1
python -u tools/train_crops3d.py configs/default_crops3d.yaml --gpu_ids 0 --num_gpus 1 


Train:
bash scripts/train_maize_full.sh
Test:
cd /work/imc_lab/c_zhang/3d-plant

  
All-crop structured-text test:
export PYTHONPATH="/home/zwh/3d-plant:/home/zwh/3d-plant/gres_model/lib:${PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
python -u tools/test_crops3d.py configs/crops3d_full_baseline_fpfix_v15_module_ablation.yaml \
  --checkpoint exps/crops3d_full_baseline_fpfix_v15_module_ablation/20261109_112220/best.pth \
  --test_mode structured \
  --sample_seed 1999 \
  --gpu_ids 0 --num_gpus 1

 

Batch visualize all texts:
python -u tools/batch_visualize_test.py configs/crops3d_full_baseline_fpfix_v15.yaml \
  --checkpoint exps/crops3d_full_baseline_fpfix_v15/20260720_182330/best.pth \
  --export_dir exps/crops3d_full_baseline_fpfix_v15/20260720_182330/vis_test_structured_pcd_3\
  --queries_json data/crops3d/all_crops_full_fpfix/crops3d_test_structured.json \
  --split test \
  --sample_seed 1999 \
  --gpu_ids 1 --num_gpus 1




---

## 1. Goal

Move MDIN from **ScanNet indoor scenes + Multi3DRefer** to **Crops3D crop point clouds + template text queries**:

```
text query (e.g. "Segment all leaves") + 3D point cloud → organ mask
```

---

## 2. Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0  tools/split_train_val.py                               │
│          Current: 7:3 train/val from local official-train scenes│
│          Official test unchanged                                │
├─────────────────────────────────────────────────────────────────┤
│  Step 1  batch_all_crops.py                                     │
│          Read PLY scalar_sf → train/val/test JSONL              │
├─────────────────────────────────────────────────────────────────┤
│  Step 2  tools/convert_jsonl_to_crops3d_json.py                 │
│          JSONL → crops3d_train/val/test.json                    │
├─────────────────────────────────────────────────────────────────┤
│  Step 3  data/crops3d/prepare_crops3d.py                        │
│          PLY → train/ val/ test/ *_refer.pth                    │
├─────────────────────────────────────────────────────────────────┤
│  Step 4  gres_model/dataset/crops3d_organ.py                    │
│          Crops3DOrganDataset loader                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 5  tools/train_crops3d.py                                 │
│          Train (val picks best.pth); test is unused in training │
├─────────────────────────────────────────────────────────────────┤
│  Step 6  tools/test_crops3d.py                                  │
│          Official test eval (run once)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Split policy (current: 7:3 train/val, official test unchanged)

| Split | Source | ~scenes | ~queries | Use |
|------|------|-------------|-------------|------|
| **train** | 70% of official-train PLY that exist locally | 576 PLY | 6454 (all-organ spatial+relation; no oversample copies) | training |
| **val** | 30% of official-train PLY that exist locally | 247 PLY | 2817 (same spatial+relation as train) | val + pick best |
| **test** | official test | 357 PLY | 1328 (canonical) | **final eval once after training** |

> **Official lists vs local PLY**: `Crops3D_10k/train.txt` may list undownloaded paths (e.g. `Rice2/`, 16 train + 8 test). Only the 8 local crop types are used.

Current split: `tools/split_train_val.py --val_ratio 0.3` on available official-train scenes, stratified by crop×organ. `data/crops3d/splits/train.txt` has 576 scenes, `val.txt` has 247. Official test is final-eval only.

**Notes:**

- Official 10k `train.txt`/`test.txt` may contain **paths that are not on disk** (e.g. undownloaded `Rice2/`); the split script skips them and records `skipped_by_crop_*` in `split_summary.json`.
- After changing the split, **rebuild query JSON and retrain**; `best.pth` is chosen by val.
- Train/val keep the same all-organ scene-graph-style structured text; test stays canonical; hand-written paraphrases and oversample copies are disabled.
- Structured text follows scene-graph organization but currently keeps only `spatial context / relation` NL queries, to cut query count and long-run load. All organs are covered; generic organs use conservative default context. No explicit `graph_node` / `graph_edge`, and MDIN / loss are unchanged.

Paper wording:

> Official Crops3D train scenes that exist locally are stratified 7:3 into train and val. Both use spatial/relation structured text. The official test set is used only after training for final evaluation.

---

## 3. One-shot run

> **Default input is full `Crops3D/Crops3D` clouds**; training downsamples with `max_npoint: 10000`. If I/O is slow, use `scripts/prepare_crops3d_data_10k.sh` (FPS 10k).

### 3.1 Data prep (Steps 0–3)

```bash
cd /mnt/e/PointCloud_zwh/MDIN-main
bash scripts/prepare_crops3d_data.sh
```

Or stepwise:

```bash
# Steps 0–3, full Crops3D by default
bash scripts/prepare_crops3d_data.sh

# or FPS 10k
bash scripts/prepare_crops3d_data_10k.sh
```

Stepwise (full):

```bash
python tools/split_train_val.py \
  --official_split_dir Crops3D/Crops3D \
  --crops_root Crops3D/Crops3D \
  --val_ratio 0.3 \
  --min_val_scenes_per_organ 2 \
  --seed 1999
python batch_all_crops.py --crops_root Crops3D/Crops3D
python tools/convert_jsonl_to_crops3d_json.py
python data/crops3d/prepare_crops3d.py --crops_root Crops3D/Crops3D --num_workers 4
```

**10k version (common in current experiments)** equals `bash scripts/prepare_crops3d_data_10k.sh`, or:

```bash
python tools/split_train_val.py \
  --official_split_dir Crops3D/Crops3D_10k \
  --crops_root Crops3D/Crops3D_10k \
  --val_ratio 0.3 \
  --min_val_scenes_per_organ 2 \
  --seed 1999
python batch_all_crops.py --crops_root Crops3D/Crops3D_10k
python tools/convert_jsonl_to_crops3d_json.py
python data/crops3d/prepare_crops3d.py --crops_root Crops3D/Crops3D_10k --num_workers 4
```

Split and query stats: `data/crops3d/splits/split_summary.json`, `data/crops3d/query_count_by_split.json`.

When full pth lives on E: drvFS, use `num_workers: 0`.

### 3.2 Training (Steps 4–5)

**Recommended (stable WSL, single GPU, no debug):**

```bash
conda activate 3d-gres
bash scripts/train_crops3d_stable.sh
```

If `save_iter > 0`, resume from `last_iter.pth`:

```bash
bash scripts/resume_crops3d_stable.sh exps/default_crops3d/YYYYMMDD_HHMMSS
```

Or the standard entry:

```bash
bash scripts/train_crops3d.sh
```

**WSL + CUDA long-run notes:**

- Do not start training as root (user `zwh` may be unable to `pkill` leftover processes)
- If hung: `pkill -9 -f train_crops3d.py`; if that fails, `wsl --shutdown` on Windows then restart WSL
- Config uses `preload_scenes: true`, `pin_memory: false`; `save_iter: 0` disables iter checkpoints to avoid WSL I/O stalls
- For real experiments set `epochs` / `lr_scheduler.max_iters` back to 100 (config may still be a 3-epoch smoke test)

### 3.3 Final test (Step 6, once after training)

```bash
bash scripts/test_crops3d.sh exps/default_crops3d/YYYYMMDD_HHMMSS/best.pth
```

---

## 4. Layout

```
MDIN-main/
├── Crops3D/
│   ├── Crops3D/                  # full PLY (default input)
│   └── Crops3D_10k/              # FPS 10k PLY (optional, ~216KB/file)
│       ├── Cabbage/*_10k.ply
│       ├── train.txt / test.txt
├── crop_label_text.py            # Step 1 core
├── batch_all_crops.py            # Step 1 entry
├── data/
│   ├── crops3d/
│   │   ├── splits/               # Step 0
│   │   │   ├── train.txt / val.txt / test.txt
│   │   ├── crops3d_train.json
│   │   ├── crops3d_val.json
│   │   ├── crops3d_test.json
│   │   ├── query_count_by_split.json
│   │   ├── train/ val/ test/     # *_refer.pth
│   └── text_queries/
│       ├── train_queries.jsonl
│       ├── val_queries.jsonl
│       └── test_queries.jsonl
├── tools/
│   ├── split_train_val.py
│   ├── convert_jsonl_to_crops3d_json.py
│   ├── train_crops3d.py
│   └── test_crops3d.py
├── scripts/
│   ├── prepare_crops3d_data.sh
│   ├── prepare_crops3d_data_10k.sh   # 30% val + stratified, FPS 10k
│   ├── train_crops3d.sh
│   ├── train_crops3d_stable.sh       # stable WSL train
│   ├── resume_crops3d_stable.sh      # resume last_iter.pth when save_iter is on
│   └── test_crops3d.sh
└── docs/
    └── CROPS3D_ADAPTATION.md     # this file
```

---

## 5. Data formats

### 5.1 JSONL (Step 1)

```json
{
  "sample_id": "Cabbage/mvs_1005_01__leaf",
  "split": "train",
  "crop": "Cabbage",
  "text": "Segment all leaves",
  "label_id": 2,
  "organ_en": "leaf",
  "point_cloud": "Crops3D/Crops3D/Cabbage/mvs_1005_01.ply"
}
```

### 5.2 MDIN annotation JSON (Step 2)

```json
{
  "scene_id": "Cabbage_mvs_1005_01",
  "ann_id": 0,
  "description": "Segment all leaves",
  "object_ids": [2],
  "object_name": "leaf",
  "crop": "Cabbage",
  "eval_type": "organ"
}
```

**Design:** `object_ids = [label_id]`. Preprocess sets `instance_label = scalar_sf`, so GT masks are organ semantic labels.

### 5.3 `_refer.pth` (Step 3)

```python
(coords, colors, superpoint, sem_labels, instance_labels)
# coords:       [N, 3] mean-centered
# colors:       [N, 3] RGB in [-1, 1]
# superpoint:   [N]    voxel-hash superpoint ID
# sem_labels:   [N]    = scalar_sf
# instance_labels: [N] = scalar_sf (organ-level)
```

### 5.4 RGB usage

Raw Crops3D PLY **has color** (`red / green / blue`). The adaptation **keeps and uses it**:

```
PLY (x,y,z, R,G,B, scalar_sf)
        ↓ prepare_crops3d.py
_refer.pth colors [N,3], normalized to [-1, 1]
        ↓ Crops3DOrganDataset
feats = concat(RGB, XYZ)  → 6-D input
        ↓ MODEL (input_channel=6)
Sparse U-Net backbone
```

| Stage | Color handling |
|------|-------------|
| preprocess | read `red/green/blue`, divide by 127.5 then subtract 1 (same as ScanNet) |
| train aug | `rgb += randn(3) * 0.1` light jitter |
| model input | with `use_xyz: True`, RGB(3) + XYZ(3) = **6 channels** |
| config | `input_channel: 6` |

Color is useful for organ segmentation, e.g.:

- **leaf** greener, **soil** browner, **fruit** red/yellow
- Complements the text query (e.g. "Segment all leaves") as **color + language**

To drop XYZ, set `use_xyz: False` and `input_channel: 3` (usually not recommended; geometry still matters).

---

## 6. Differences from original MDIN

| Item | Original MDIN | Crops3D adaptation |
|------|-----------|--------------|
| clouds | ScanNet indoor | Crops3D crops (**with RGB**) |
| text | free-form NL | train/val: templates + spatial/relation; test: canonical |
| GT mask | instance ID | organ semantic label |
| scene graph | sng_parser | **unused** |
| contrastive align | on | **off** |
| ScanNet pretrain | used | **unused** |
| metrics | zt/mt/st subsets | mIoU by crop / organ |

---

## 7. Config highlights

Key fields in `configs/default_crops3d.yaml`:

```yaml
data:
  train:
    type: crops3d_organ
    max_npoint: 6000      # current WSL-stable setting; tune for VRAM
    preload_scenes: true  # preload scenes; fewer WSL I/O hangs
    with_elastic: false   # elastic off (order bug is fixed; re-enable if needed)
dataloader:
  train:
    batch_size: 1         # large clouds: 1
    pin_memory: false     # recommended on WSL
model:
  pretrain: null          # no ScanNet weights
  fix_module: []          # no frozen modules
  contrastive_align_loss: False
train:
  save_iter: 0            # WSL-stable: no iter checkpoints
  early_stop:
    enabled: true
    patience: 10            # stop after 10 val rounds with no gain
    metric: mIOU            # same as best.pth
```

---

## 8. Metrics

`train_crops3d.py` reports:

- **Overall mIoU** / Acc@0.25 / Acc@0.5
- **Macro crop mIoU** / **Macro organ mIoU**
- **Per-crop mIoU / Acc@0.5 / Acc@0.25** (8 crops)
- **Per-organ mIoU / Acc@0.5 / Acc@0.25** (leaf / stem / fruit, etc.)
- **Worst samples by IoU** (error analysis)

TensorBoard: `val/macro_crop_mIOU`, `val/macro_organ_mIOU`, `val/crop_{Crop}_*`, `val/organ_{organ}_*`

---

## 9. Dependencies

Data prep:

```bash
pip install plyfile numpy tqdm
```

Training (same as original MDIN):

```bash
# PyTorch, spconv, pointgroup_ops, torch_scatter, transformers, ...
```

---

## 10. FAQ

### Q1: `no valid samples`

Confirm Step 3 ran and `data/crops3d/train/` contains `{Crop}_{filename}_refer.pth`.

### Q2: OOM

- Lower `max_npoint` (e.g. 50000)
- Keep `batch_size: 1`
- Keep `with_elastic: False`

### Q3: ScanNet pretrain?

Not recommended. Indoor vs crop domain gap is large. If you need pretrain, train the backbone on Crops3D first.

### Q4: Richer text templates?

Edit `label_map` text in `CROP_CONFIG` inside `crop_label_text.py`, e.g.:

```python
2: ("leaf", "the green leaves of the plant"),
```

Structured text is written into **train/val**; test stays canonical. After changing templates or splits, rerun Steps 0–3.

### Q5: Val noisy / few Rice samples?

Already using **30% val + crop×organ stratification** (see §2.1). Rice official train is small; current val is 17 scenes / 102 queries (with structured text). When picking best, also look at per-crop mIoU or macro_crop_mIoU.

### Q6: Can old checkpoints still be used?

**Not** for direct comparison. Since 2026-06 the split is 7:3 train/val and val text matches train. Rebuild data and **train from scratch** (or at least re-select best on the new val).

---

## 11. Suggested paper experiments

| Experiment | Note |
|------|------|
| Baseline-semantic | no text; segment by label |
| Baseline-PointNet | classic point-cloud segmentation |
| **Ours-MDIN** | text query + MDIN |
| Ablation-text | templates A/B/C |
| Ablation-loss | contrastive align on/off |
| Cross-crop | leave-one-crop-out |

---

## 12. Changed files

| File | Action | Note |
|------|------|------|
| `tools/split_train_val.py` | rewrite | Step 0: 30% val + crop×organ; skip missing local PLY |
| `batch_all_crops.py` | edit | use splits dir |
| `crop_label_text.py` | edit | train/val/test, rare-organ oversample, scene-graph-style text |
| `tools/convert_jsonl_to_crops3d_json.py` | edit | three JSON files |
| `data/crops3d/prepare_crops3d.py` | edit | train/val/test dirs |
| `configs/default_crops3d.yaml` | edit | val uses val/, preload, save_iter, ... |
| `tools/train_crops3d.py` | edit | optional save_iter resume, scan logs |
| `tools/test_crops3d.py` | new | official test eval |
| `scripts/prepare_crops3d_data*.sh` | edit | val_ratio=0.3, min_val_scenes_per_organ=2 |
| `scripts/train_crops3d_stable.sh` | new | stable WSL train |
| `scripts/resume_crops3d_stable.sh` | new | last_iter resume when save_iter is on |
| `data/crops3d/query_count_by_split.json` | update | query stats for the new split |
| `docs/CROPS3D_ADAPTATION.md` | edit | this file |

**Important:** After the 2026-06 change to 7:3 train/val + stratified split and val text aligned with train, **rerun Steps 0–3 and retrain**. Older runs such as `20260621_230618` (10% val) are historical only.

---

## 13. Follow-ups

- [ ] Richer NL templates
- [ ] Finer superpoints via mesh/segmentator
- [ ] Cross-crop zero-shot eval
- [ ] Compare with ELGCot3D and other segmentation baselines
- [ ] Add a standalone `tools/test_crops3d.py` test script
