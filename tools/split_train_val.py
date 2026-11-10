"""Step 0: split val from official train; crop x organ stratify + min val scenes."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crop_label_text import get_crop_config, load_labels_from_ply, normalize_label_map_organ


def read_split_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def extract_crop_from_path(line):
    return Path(line.replace("\\", "/")).parent.name


def resolve_ply_path(project_root, line, crops_root=None):
    """Resolve a train.txt line to a local PLY path (data/Crops3D_10k or Crops3D/Crops3D_10k)."""
    norm = line.replace("\\", "/")
    rel = Path(norm)
    candidates = [project_root / norm]
    if crops_root is not None:
        crops_root = Path(crops_root)
        if not crops_root.is_absolute():
            crops_root = project_root / crops_root
        candidates.append(crops_root / rel.name)
        candidates.append(crops_root / rel.parent.name / rel.name)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def filter_existing_ply_lines(lines, project_root, crops_root):
    """Skip official-split paths with no local PLY (e.g. undownloaded Rice2)."""
    kept = []
    skipped = []
    for line in lines:
        ply_path = resolve_ply_path(project_root, line, crops_root=crops_root)
        if ply_path.exists():
            kept.append(line)
        else:
            skipped.append(
                {
                    "line": line,
                    "crop": extract_crop_from_path(line),
                    "resolved": str(ply_path),
                }
            )
    skipped_by_crop = defaultdict(int)
    for item in skipped:
        skipped_by_crop[item["crop"]] += 1
    return kept, skipped, dict(skipped_by_crop)


def get_crop_config_safe(crop):
    try:
        return get_crop_config(crop)
    except KeyError:
        return None


def get_all_organs_for_crop(crop):
    config = get_crop_config_safe(crop)
    if config is None:
        return set()
    organs = {normalize_label_map_organ(v) for v in config["label_map"].values()}
    for comp in config.get("composite_organs", []):
        organs.add(comp["organ_en"])
    return organs


def get_scene_organs(ply_path, crop):
    """Infer query organs for a scene from PLY scalar_sf."""
    config = get_crop_config_safe(crop)
    if config is None:
        return set()
    if not ply_path.exists():
        return get_all_organs_for_crop(crop)

    labels = load_labels_from_ply(ply_path)
    present = set(labels.tolist())
    exclude = config.get("exclude_labels", set())

    organs = set()
    for label_id, organ_en in ((lid, normalize_label_map_organ(v)) for lid, v in config["label_map"].items()):
        if label_id in present and label_id not in exclude:
            organs.add(organ_en)
    for comp in config.get("composite_organs", []):
        if any(int(lid) in present for lid in comp["label_ids"]):
            organs.add(comp["organ_en"])
    return organs


def build_scene_records(lines, project_root, crops_root):
    records = []
    for line in lines:
        crop = extract_crop_from_path(line)
        ply_path = resolve_ply_path(project_root, line, crops_root=crops_root)
        organs = get_scene_organs(ply_path, crop)
        records.append({"line": line, "crop": crop, "organs": organs, "ply": str(ply_path)})
    return records


def split_crop_scenes(crop_scenes, val_ratio, min_val_scenes_per_organ, rng):
    """Within a crop: meet min val scenes per organ, then approach val_ratio."""
    n = len(crop_scenes)
    if n == 0:
        return [], []
    if n == 1:
        return [], [crop_scenes[0]["line"]]

    target_val = max(1, int(round(n * val_ratio)))
    val_lines = set()
    organ_val_count = defaultdict(int)

    all_organs = set()
    for scene in crop_scenes:
        all_organs |= scene["organs"]

    pool = crop_scenes.copy()
    rng.shuffle(pool)

    # Phase 1: at least min_val_scenes_per_organ scenes per crop x organ into val
    for organ in sorted(all_organs):
        while organ_val_count[organ] < min_val_scenes_per_organ:
            picked = None
            for scene in pool:
                if scene["line"] in val_lines:
                    continue
                if organ in scene["organs"]:
                    picked = scene
                    break
            if picked is None:
                break
            val_lines.add(picked["line"])
            for o in picked["organs"]:
                organ_val_count[o] += 1

    # Phase 2: fill to target_val if short; keep Phase 1 if already over
    remaining = [s for s in pool if s["line"] not in val_lines]
    rng.shuffle(remaining)
    while len(val_lines) < target_val and remaining:
        scene = remaining.pop()
        val_lines.add(scene["line"])
        for o in scene["organs"]:
            organ_val_count[o] += 1

    train_lines = [s["line"] for s in crop_scenes if s["line"] not in val_lines]
    val_out = [s["line"] for s in crop_scenes if s["line"] in val_lines]
    return train_lines, val_out, dict(organ_val_count)


def stratified_split_crop_organ(lines, project_root, crops_root, val_ratio, min_val_scenes_per_organ, seed):
    rng = random.Random(seed)
    records = build_scene_records(lines, project_root, crops_root)

    by_crop = defaultdict(list)
    for rec in records:
        by_crop[rec["crop"]].append(rec)

    train_lines, val_lines = [], []
    organ_val_coverage = {}
    crop_stats = {}

    for crop in sorted(by_crop.keys()):
        crop_train, crop_val, organ_counts = split_crop_scenes(
            by_crop[crop], val_ratio, min_val_scenes_per_organ, rng
        )
        train_lines.extend(crop_train)
        val_lines.extend(crop_val)
        organ_val_coverage[crop] = organ_counts
        crop_stats[crop] = {
            "num_train_scenes": len(crop_train),
            "num_val_scenes": len(crop_val),
            "organ_val_scene_counts": organ_counts,
            "has_queries": get_crop_config_safe(crop) is not None,
        }

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines, crop_stats, organ_val_coverage


def write_split_file(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split val from official train (crop x organ, default 30% val)"
    )
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument(
        "--official_split_dir",
        type=str,
        default="Crops3D/Crops3D_10k",
        help="dir with official train.txt / test.txt",
    )
    parser.add_argument(
        "--crops_root",
        type=str,
        default=None,
        help="actual PLY root (default same as official_split_dir)",
    )
    parser.add_argument("--output_dir", type=str, default="data/crops3d/splits")
    parser.add_argument("--val_ratio", type=float, default=0.3, help="val scene ratio")
    parser.add_argument(
        "--min_val_scenes_per_organ",
        type=int,
        default=2,
        help="min val scenes per crop x organ (use as many as exist if fewer)",
    )
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    official_dir = project_root / args.official_split_dir
    crops_root = args.crops_root or args.official_split_dir
    output_dir = project_root / args.output_dir

    official_train = official_dir / "train.txt"
    if not official_train.exists():
        raise FileNotFoundError(f"official train split not found: {official_train}")

    train_lines = read_split_lines(official_train)
    train_available, skipped_train, skipped_by_crop = filter_existing_ply_lines(
        train_lines, project_root, crops_root
    )
    train_split, val_split, crop_stats, organ_val_coverage = stratified_split_crop_organ(
        train_available,
        project_root,
        crops_root,
        val_ratio=args.val_ratio,
        min_val_scenes_per_organ=args.min_val_scenes_per_organ,
        seed=args.seed,
    )

    write_split_file(output_dir / "train.txt", train_split)
    write_split_file(output_dir / "val.txt", val_split)

    official_test = official_dir / "test.txt"
    skipped_test = []
    if official_test.exists():
        test_lines_raw = read_split_lines(official_test)
        test_lines, skipped_test, skipped_test_by_crop = filter_existing_ply_lines(
            test_lines_raw, project_root, crops_root
        )
        write_split_file(output_dir / "test.txt", test_lines)
    else:
        test_lines = []
        skipped_test_by_crop = {}
        print(f"warn: missing {official_test}")

    summary = {
        "val_ratio": args.val_ratio,
        "min_val_scenes_per_organ": args.min_val_scenes_per_organ,
        "seed": args.seed,
        "strategy": "stratified_crop_x_organ",
        "num_official_train": len(train_lines),
        "num_train_ply_available": len(train_available),
        "num_skipped_missing_ply_train": len(skipped_train),
        "num_skipped_missing_ply_test": len(skipped_test),
        "skipped_by_crop_train": skipped_by_crop,
        "skipped_by_crop_test": skipped_test_by_crop,
        "num_train": len(train_split),
        "num_val": len(val_split),
        "num_test": len(test_lines),
        "crop_stats": crop_stats,
        "organ_val_coverage": organ_val_coverage,
        "note": (
            "train/val come from official-train PLY that exist locally; after grouping by crop, "
            "enforce min val scenes per crop x organ, then approach val_ratio. test is local official-test PLY, unused in training."
        ),
    }
    if skipped_train or skipped_test:
        summary["skipped_examples"] = {
            "train": skipped_train[:5],
            "test": skipped_test[:5],
        }
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Split done (crop x organ):")
    print(f"  official train : {len(train_lines)}")
    if skipped_train:
        print(
            f"  skipped missing PLY: {len(skipped_train)} "
            f"({', '.join(f'{k}={v}' for k, v in sorted(skipped_by_crop.items()))})"
        )
    print(f"  available train : {len(train_available)}")
    denom = len(train_available) or 1
    print(f"  -> train   : {len(train_split)}  ({100 * len(train_split) / denom:.1f}%)")
    print(f"  -> val     : {len(val_split)}  ({100 * len(val_split) / denom:.1f}%)")
    print(f"  -> test    : {len(test_lines)}", end="")
    if skipped_test:
        print(f"  (skipped missing {len(skipped_test)})", end="")
    print()
    print(f"  output dir    : {output_dir}")
    print("\nval scenes per crop:")
    for crop, st in sorted(crop_stats.items()):
        print(f"  {crop:12s} train={st['num_train_scenes']:3d}  val={st['num_val_scenes']:3d}")
    print("\nNext: python batch_all_crops.py --crops_root", crops_root)


if __name__ == "__main__":
    main()
