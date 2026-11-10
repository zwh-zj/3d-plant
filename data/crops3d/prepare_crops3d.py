"""Step 3: Crops3D PLY -> train/val/test *_refer.pth.

Each file stores:
(coords, colors, superpoint, sem_labels, instance_labels)
coords are mean-centered; colors are RGB/127.5-1;
superpoint is a voxel-hash ID; sem_labels and instance_labels equal scalar_sf.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def voxel_hash_superpoint(coords, voxel_size):
    """Quantize coords by voxel size and unique them to get per-point superpoint IDs."""
    discrete = np.floor(coords / float(voxel_size)).astype(np.int64)
    discrete -= discrete.min(axis=0)
    _, superpoint = np.unique(discrete, axis=0, return_inverse=True)
    return superpoint.astype(np.int64)


def process_ply(ply_path, output_path, voxel_size=0.03):
    """Read one PLY, write the matching *_refer.pth, return the output filename."""
    ply_path = Path(ply_path)
    output_path = Path(output_path)
    vertex = PlyData.read(str(ply_path))["vertex"]
    coords = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float32)
    coords -= coords.mean(axis=0, dtype=np.float32)
    colors = np.stack(
        [vertex["red"], vertex["green"], vertex["blue"]], axis=1
    ).astype(np.float32)
    colors = colors / 127.5 - 1.0
    labels = np.rint(np.asarray(vertex["scalar_sf"], dtype=np.float64)).astype(np.int64)
    superpoint = voxel_hash_superpoint(coords, voxel_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((coords, colors, superpoint, labels, labels.copy()), output_path)
    return output_path.name


def _worker(task):
    ply_path, output_path, voxel_size = task
    try:
        name = process_ply(ply_path, output_path, voxel_size)
        return {"status": "ok", "file": name}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "file": str(ply_path), "error": str(exc)}


def resolve_ply(rel_ply, crops_root):
    rel = rel_ply.replace("\\", "/")
    name = Path(rel).name
    crop = Path(rel).parent.name
    candidates = [
        crops_root / crop / name,
        crops_root / name,
        PROJECT_ROOT / rel,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def collect_tasks(annotation_dir, crops_root, output_root, voxel_size, skip_existing):
    tasks = []
    skipped = 0
    missing = []
    seen = set()
    for json_name in ("crops3d_train.json", "crops3d_val.json", "crops3d_test.json"):
        path = annotation_dir / json_name
        if not path.exists():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        for entry in entries:
            scene_id = entry["scene_id"]
            if scene_id in seen:
                continue
            seen.add(scene_id)
            split = entry.get("split", "train")
            out_path = output_root / split / f"{scene_id}_refer.pth"
            if skip_existing and out_path.exists():
                skipped += 1
                continue
            ply_path = resolve_ply(entry["point_cloud"], crops_root)
            if not ply_path.exists():
                missing.append(str(ply_path))
                continue
            tasks.append((str(ply_path), str(out_path), float(voxel_size)))
    return tasks, skipped, missing


def main():
    parser = argparse.ArgumentParser(description="Crops3D PLY -> *_refer.pth")
    parser.add_argument("--crops_root", type=str, default="Crops3D/Crops3D")
    parser.add_argument("--annotation_dir", type=str, default="data/crops3d")
    parser.add_argument("--output_root", type=str, default="data/crops3d")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--voxel_size", type=float, default=0.03)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    crops_root = Path(args.crops_root)
    if not crops_root.is_absolute():
        crops_root = PROJECT_ROOT / crops_root
    annotation_dir = Path(args.annotation_dir)
    if not annotation_dir.is_absolute():
        annotation_dir = PROJECT_ROOT / annotation_dir
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    tasks, skipped, missing = collect_tasks(
        annotation_dir, crops_root, output_root, args.voxel_size, args.skip_existing
    )
    print(
        f"crops_root={crops_root}\n"
        f"annotation_dir={annotation_dir}\n"
        f"output_root={output_root}\n"
        f"voxel_size={args.voxel_size} workers={args.num_workers}\n"
        f"skip_existing={skipped} missing_ply={len(missing)} todo={len(tasks)}",
        flush=True,
    )
    if missing:
        print("sample missing", missing[:5], flush=True)
    if not tasks:
        print("nothing to do", flush=True)
        return

    errors = []
    workers = max(1, int(args.num_workers))
    if workers == 1:
        results = map(_worker, tasks)
        iterator = enumerate(results, 1)
        for i, result in iterator:
            if result["status"] == "error":
                errors.append(result)
                print("ERR", result, flush=True)
            if i % 20 == 0 or i == len(tasks):
                print(f"progress {i}/{len(tasks)} errors={len(errors)}", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, tasks), 1):
                if result["status"] == "error":
                    errors.append(result)
                    print("ERR", result, flush=True)
                if i % 20 == 0 or i == len(tasks):
                    print(f"progress {i}/{len(tasks)} errors={len(errors)}", flush=True)

    print(
        f"done success={len(tasks) - len(errors)} errors={len(errors)} skipped={skipped}",
        flush=True,
    )
    if errors:
        print("sample errors", errors[:5], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
