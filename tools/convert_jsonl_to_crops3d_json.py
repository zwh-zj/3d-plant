"""Step 2: convert JSONL queries to MDIN crops3d JSON (train/val/test)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    """Read a JSONL file, one JSON object per line."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def ply_stem_to_scene_id(crop, point_cloud_path):
    """
    Build scene_id from a relative PLY path.
    Example: Crops3D/Crops3D/Cabbage/mvs_1005_01.ply -> Cabbage_mvs_1005_01
    """
    stem = Path(point_cloud_path.replace("\\", "/")).stem
    return f"{crop}_{stem}"


def convert_entries(entries):
    """
    Convert a JSONL item to MDIN format.
    ann_id increments from 0 within the same scene (same PLY).
    """
    scene_ann_counter = defaultdict(int)
    converted = []

    for entry in entries:
        crop = entry["crop"]
        scene_id = ply_stem_to_scene_id(crop, entry["point_cloud"])
        ann_id = scene_ann_counter[scene_id]
        scene_ann_counter[scene_id] += 1

        # organ seg: single label or virtual compound (stem+leaf -> shoot_system)
        if "label_ids" in entry:
            object_ids = [int(x) for x in entry["label_ids"]]
        else:
            object_ids = [entry["label_id"]]
        converted.append(
            {
                "scene_id": scene_id,
                "ann_id": ann_id,
                "description": entry["text"],
                "object_ids": object_ids,
                "object_name": entry["organ_en"],
                "crop": crop,
                "eval_type": "organ",
                "spatial": False,
                "color": False,
                "texture": False,
                "shape": False,
                "split": entry.get("split", "unknown"),
                "point_cloud": entry["point_cloud"],
                "sample_id": entry.get("sample_id", ""),
                "point_count": entry.get("point_count", 0),
                "text_variant": entry.get(
                    "text_variant",
                    "canonical" if not entry.get("is_paraphrase") else "structured",
                ),
                "canonical_sample_id": entry.get("canonical_sample_id", entry.get("sample_id", "")),
            }
        )

    return converted


def save_json(entries, path):
    """Write a JSON annotation file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  {len(entries):5d} items -> {path}")


def main():
    parser = argparse.ArgumentParser(description="JSONL -> crops3d_train/val/test.json")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="project root",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="data/text_queries",
        help="JSONL input dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/crops3d",
        help="JSON output dir",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    jsonl_dir = project_root / args.jsonl_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # three JSONL files -> three JSON files
    split_pairs = [
        ("train_queries.jsonl", "crops3d_train.json"),
        ("val_queries.jsonl", "crops3d_val.json"),
        ("test_queries.jsonl", "crops3d_test.json"),
        ("test_structured_queries.jsonl", "crops3d_test_structured.json"),
    ]

    counts = {}
    entries_by_split = {}
    for jsonl_name, json_name in split_pairs:
        jsonl_path = jsonl_dir / jsonl_name
        if not jsonl_path.exists():
            print(f"warn: missing {jsonl_path}, skip")
            counts[json_name] = 0
            entries_by_split[json_name] = []
            continue
        entries = convert_entries(load_jsonl(jsonl_path))
        save_json(entries, output_dir / json_name)
        counts[json_name] = len(entries)
        entries_by_split[json_name] = entries

    if counts.get("crops3d_train.json", 0) == 0:
        raise FileNotFoundError(
            f"training JSONL not found; run: python batch_all_crops.py"
        )

    # stats from the training set
    train_path = output_dir / "crops3d_train.json"
    with open(train_path, "r", encoding="utf-8") as f:
        train_entries = json.load(f)

    crop_stats = defaultdict(int)
    organ_stats = defaultdict(int)
    for e in train_entries:
        crop_stats[e["crop"]] += 1
        organ_stats[e["object_name"]] += 1

    split_note = (
        "train/val are a 7:3 stratified split of official train scenes that exist locally; "
        "train and val use spatial+relation structured text; "
        "crops3d_test.json is official test canonical templates; "
        "crops3d_test_structured.json is structured-text test on the same scenes/GT"
    )
    summary = {
        "num_train": counts.get("crops3d_train.json", 0),
        "num_val": counts.get("crops3d_val.json", 0),
        "num_test": counts.get("crops3d_test.json", 0),
        "num_test_structured": counts.get("crops3d_test_structured.json", 0),
        "num_scenes_train": len(set(e["scene_id"] for e in train_entries)),
        "split_note": split_note,
        "crop_distribution": dict(sorted(crop_stats.items())),
        "organ_distribution": dict(sorted(organ_stats.items(), key=lambda x: -x[1])),
    }
    summary_path = output_dir / "annotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    query_stats = {
        "note": split_note,
        "totals": {},
    }
    split_names = {
        "train": "crops3d_train.json",
        "val": "crops3d_val.json",
        "test": "crops3d_test.json",
        "test_structured": "crops3d_test_structured.json",
    }
    for split_name, json_name in split_names.items():
        entries = entries_by_split.get(json_name, [])
        crop_counts = defaultdict(int)
        organ_counts = defaultdict(int)
        for entry in entries:
            crop_counts[entry["crop"]] += 1
            organ_counts[entry["object_name"]] += 1

        query_stats["totals"][split_name] = {
            "queries": len(entries),
            "scenes": len(set(entry["scene_id"] for entry in entries)),
        }
        query_stats[f"by_crop_{split_name}"] = dict(sorted(crop_counts.items()))
        query_stats[f"by_organ_{split_name}"] = dict(
            sorted(organ_counts.items(), key=lambda item: -item[1])
        )

    split_summary_path = output_dir / "splits" / "split_summary.json"
    if split_summary_path.exists():
        with open(split_summary_path, "r", encoding="utf-8") as f:
            split_summary = json.load(f)
        query_stats["split"] = {
            "val_ratio": split_summary.get("val_ratio"),
            "strategy": split_summary.get("strategy"),
            "num_train": split_summary.get("num_train"),
            "num_val": split_summary.get("num_val"),
            "num_test": split_summary.get("num_test"),
            "test_protocol": "official_test_unchanged",
        }

    query_stats_path = output_dir / "query_count_by_split.json"
    with open(query_stats_path, "w", encoding="utf-8") as f:
        json.dump(query_stats, f, ensure_ascii=False, indent=2)
    print(f"\nStats -> {summary_path}")
    print(f"Per-split stats -> {query_stats_path}")
    print("Next: python data/crops3d/prepare_crops3d.py")


if __name__ == "__main__":
    main()
