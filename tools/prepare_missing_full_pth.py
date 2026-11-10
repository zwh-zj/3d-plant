#!/usr/bin/env python3
"""Preprocess full PLY only for missing *_refer.pth listed in annotation JSON; skip existing files."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data" / "crops3d"))
from prepare_crops3d import process_ply  # noqa: E402


def worker(task):
    ply_path, output_path, voxel_size = task
    try:
        name = process_ply(Path(ply_path), Path(output_path), voxel_size)
        return {"status": "ok", "file": name}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "file": str(ply_path), "error": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_dir", default="data/crops3d/all_crops_full")
    p.add_argument("--crops_root", default="Crops3D/Crops3D")
    p.add_argument("--output_root", default="data/crops3d")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--voxel_size", type=float, default=0.03)
    args = p.parse_args()

    anno = PROJECT_ROOT / args.annotation_dir
    crops_root = PROJECT_ROOT / args.crops_root
    output_root = PROJECT_ROOT / args.output_root

    all_ply = {}
    for jf in ("crops3d_train.json", "crops3d_val.json", "crops3d_test.json"):
        path = anno / jf
        if not path.exists():
            continue
        for e in json.loads(path.read_text(encoding="utf-8")):
            all_ply[e["scene_id"]] = (e["point_cloud"], e.get("split", "train"))

    tasks = []
    skipped = 0
    for scene_id, (rel_ply, split) in sorted(all_ply.items()):
        out_path = output_root / split / f"{scene_id}_refer.pth"
        if out_path.exists():
            skipped += 1
            continue
        rel = rel_ply.replace("\\", "/")
        ply_name = Path(rel).name
        crop_name = Path(rel).parent.name
        ply_path = crops_root / crop_name / ply_name
        if not ply_path.exists():
            alt = PROJECT_ROOT / rel
            if alt.exists():
                ply_path = alt
            else:
                print(f"MISSING PLY {ply_path}")
                continue
        tasks.append((str(ply_path), str(out_path), args.voxel_size))

    print(f"skip_existing={skipped} todo={len(tasks)}", flush=True)
    errors = []
    if not tasks:
        print("nothing to do")
        return

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, tasks), 1):
            if result["status"] == "error":
                errors.append(result)
                print("ERR", result, flush=True)
            if i % 20 == 0 or i == len(tasks):
                print(f"progress {i}/{len(tasks)} errors={len(errors)}", flush=True)

    print(f"done success={len(tasks) - len(errors)} errors={len(errors)}", flush=True)
    if errors:
        print("sample errors", errors[:5], flush=True)


if __name__ == "__main__":
    main()
