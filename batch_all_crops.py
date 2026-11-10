"""Generate English text queries for all 8 crops (Step 1)."""

import argparse
from pathlib import Path

from crop_label_text import CROP_NAMES, batch_save_all_crops

# project root
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CROPS_ROOT = PROJECT_ROOT / "Crops3D" / "Crops3D"
SPLIT_DIR = PROJECT_ROOT / "data" / "crops3d" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "text_queries"


def main():
    parser = argparse.ArgumentParser(description="Generate Crops3D text-query JSONL")
    parser.add_argument(
        "--crops_root",
        type=str,
        default=str(DEFAULT_CROPS_ROOT),
        help="PLY root (default: full Crops3D clouds)",
    )
    args = parser.parse_args()

    crops_root = Path(args.crops_root)
    if not crops_root.is_absolute():
        crops_root = PROJECT_ROOT / crops_root

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Crops root   : {crops_root}")
    print(f"Split dir    : {SPLIT_DIR}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print(f"Crops        : {', '.join(CROP_NAMES)}\n")

    if not crops_root.exists():
        raise FileNotFoundError(
            f"Crops3D data dir not found: {crops_root}\n"
            "Place PLY files under Crops3D/Crops3D/."
        )
    if not (SPLIT_DIR / "train.txt").exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_DIR / 'train.txt'}\n"
            "Run first: python tools/split_train_val.py --official_split_dir Crops3D/Crops3D"
        )

    result = batch_save_all_crops(
        crops_root=crops_root,
        output_root=OUTPUT_ROOT,
        project_root=PROJECT_ROOT,
        split_dir=SPLIT_DIR,
    )

    s = result["summary"]
    print("\n========== Summary ==========")
    for crop, info in s["crops"].items():
        print(
            f"{crop:10s}  PLY={info['num_point_clouds']:3d}  "
            f"queries={info['num_queries']:4d}  "
            f"(train={info['num_train_queries']}, val={info['num_val_queries']}, "
            f"test={info['num_test_queries']})"
        )
    print("-----------------------------")
    print(f"Total queries : {s['num_queries']}")
    print(f"Train queries : {s['num_train_queries']}")
    print(f"Val queries   : {s['num_val_queries']}")
    print(f"Test queries  : {s['num_test_queries']}")
    print(f"Test structured: {s.get('num_test_structured_queries', 0)}")
    print(f"Errors        : {s['num_errors']}")
    print(f"\nOutput        : {result['all_jsonl_path']}")
    print("\nNext: python tools/convert_jsonl_to_crops3d_json.py")


if __name__ == "__main__":
    main()
