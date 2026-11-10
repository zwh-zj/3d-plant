"""Build a Maize-only training set (separate dir; does not modify existing files).

Output:
  data/text_queries/single_maize/          JSONL
  data/crops3d/single_maize/               crops3d_train/val/test.json
  data/crops3d/single_maize/dataset_summary.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import crop_label_text as clt
from crop_label_text import batch_save_text_queries, get_crop_config, write_jsonl
from tools.convert_jsonl_to_crops3d_json import convert_entries, load_jsonl, save_json

CROP = "Maize"

# target query count per scene x organ (canonical + spatial/relation paraphrase + oversample copies)
# default 4 paraphrases; factor=8 keeps 8 items; rare organs get a larger factor.
MAIZE_TRAIN_OVERSAMPLE_FACTORS = {
    "leaf": 8,
    "stem": 8,
    "shoot_system": 8,
    "soil": 8,
    "pot": 8,
    "tassel": 40,
    "corn": 50,
}

# 1) fpfix: stem/soil 16 (or duplicate from baseline JSON; see single_maize_full_fpfix)
MAIZE_TRAIN_OVERSAMPLE_FACTORS_FPFIX = {
    "leaf": 8,
    "stem": 16,
    "shoot_system": 8,
    "soil": 16,
    "pot": 8,
    "tassel": 40,
    "corn": 50,
}


def get_maize_oversample_factor(crop, organ_en):
    if crop != CROP:
        return 1
    return max(1, int(MAIZE_TRAIN_OVERSAMPLE_FACTORS.get(organ_en, 8)))


def convert_jsonl_dir(jsonl_dir, output_dir):
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
            counts[json_name] = 0
            entries_by_split[json_name] = []
            continue
        entries = convert_entries(load_jsonl(jsonl_path))
        save_json(entries, output_dir / json_name)
        counts[json_name] = len(entries)
        entries_by_split[json_name] = entries
    return counts, entries_by_split


def build_summary(counts, entries_by_split, jsonl_dir, output_dir, crops_root, split_dir, config_path):
    summary = {
        "crop": CROP,
        "note": (
            "Maize-only set; train has spatial/relation text and oversampling; "
            "val has paraphrases without oversampling; test is canonical templates."
        ),
        "oversample_factors": MAIZE_TRAIN_OVERSAMPLE_FACTORS,
        "paths": {
            "jsonl_dir": str(jsonl_dir.relative_to(PROJECT_ROOT)),
            "annotation_dir": str(output_dir.relative_to(PROJECT_ROOT)),
            "data_root": "data/crops3d/",
            "config": config_path,
        },
        "counts": counts,
        "totals": {},
        "by_organ": {},
        "by_organ_scene": {},
    }

    split_key = {
        "crops3d_train.json": "train",
        "crops3d_val.json": "val",
        "crops3d_test.json": "test",
    }
    for json_name, split in split_key.items():
        entries = entries_by_split.get(json_name, [])
        organ_counts = defaultdict(int)
        organ_scenes = defaultdict(set)
        for entry in entries:
            organ = entry["object_name"]
            organ_counts[organ] += 1
            organ_scenes[organ].add(entry["scene_id"])
        summary["totals"][split] = {
            "queries": len(entries),
            "scenes": len({e["scene_id"] for e in entries}),
        }
        summary["by_organ"][split] = dict(sorted(organ_counts.items()))
        summary["by_organ_scene"][split] = {
            k: len(v) for k, v in sorted(organ_scenes.items())
        }

    summary["source"] = {
        "crops_root": str(crops_root),
        "split_dir": str(split_dir),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build Maize-only training data (separate dir)")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(PROJECT_ROOT),
    )
    parser.add_argument(
        "--crops_root",
        type=str,
        default="Crops3D/Crops3D_10k",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="data/crops3d/splits",
    )
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="data/text_queries/single_maize",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/crops3d/single_maize",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="train config path (written into dataset_summary; inferred from output_dir)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    crops_root = project_root / args.crops_root
    split_dir = project_root / args.split_dir
    jsonl_dir = project_root / args.jsonl_dir
    output_dir = project_root / args.output_dir
    if args.config:
        config_path = args.config
    elif "full" in output_dir.name:
        config_path = "configs/maize_only_full.yaml"
    else:
        config_path = "configs/maize_only.yaml"
    train_script = (
        "bash scripts/train_maize_full.sh"
        if "full" in output_dir.name
        else "bash scripts/train_maize_only.sh"
    )

    maize_ply_dir = crops_root / CROP
    if not maize_ply_dir.exists():
        raise FileNotFoundError(f"Maize PLY dir not found: {maize_ply_dir}")
    if not (split_dir / "train.txt").exists():
        raise FileNotFoundError(f"split file not found: {split_dir / 'train.txt'}")

    jsonl_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_crop_config(CROP)
    original_fn = clt.get_train_oversample_factor
    clt.get_train_oversample_factor = get_maize_oversample_factor
    try:
        print(f"Generate {CROP} JSONL -> {jsonl_dir}")
        print(f"Oversample factors: {MAIZE_TRAIN_OVERSAMPLE_FACTORS}")
        result = batch_save_text_queries(
            input_dir=maize_ply_dir,
            output_dir=jsonl_dir,
            label_map=cfg["label_map"],
            exclude_labels=cfg["exclude_labels"],
            crop=CROP,
            project_root=project_root,
            split_dir=split_dir,
            composite_organs=cfg.get("composite_organs", []),
        )
    finally:
        clt.get_train_oversample_factor = original_fn

    write_jsonl(jsonl_dir / "all_queries.jsonl", result["all_entries"])
    write_jsonl(jsonl_dir / "train_queries.jsonl", result["train_entries"])
    write_jsonl(jsonl_dir / "val_queries.jsonl", result["val_entries"])
    write_jsonl(jsonl_dir / "test_queries.jsonl", result["test_entries"])
    write_jsonl(
        jsonl_dir / "test_structured_queries.jsonl",
        result["test_structured_entries"],
    )

    print(f"\nConvert JSONL -> {output_dir}")
    counts, entries_by_split = convert_jsonl_dir(jsonl_dir, output_dir)

    summary = build_summary(
        counts, entries_by_split, jsonl_dir, output_dir, crops_root, split_dir, config_path
    )
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Maize-only dataset ==========")
    for split in ["train", "val", "test"]:
        totals = summary["totals"][split]
        print(f"  {split:5s}: scenes={totals['scenes']:3d}  queries={totals['queries']:5d}")
    print(f"\n  JSONL : {jsonl_dir}")
    print(f"  JSON  : {output_dir}")
    print(f"  stats  : {summary_path}")
    print(f"\nTrain command:")
    print(f"  {train_script}")
    print(f"  # or")
    print(
        f"  python -u tools/train_crops3d.py {config_path} "
        f"--gpu_ids 0 --num_gpus 1"
    )


if __name__ == "__main__":
    main()
