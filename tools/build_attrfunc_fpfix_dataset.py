#!/usr/bin/env python3
"""Build all-crop attribute+functional labels (control group E) into an fpfix dir.

Does not modify data/crops3d/all_crops_full_fpfix/ or crop_label_text.py.

Pipeline:
  1. batch_save_all_crops_attrfunc → JSONL
  2. JSONL → crops3d_train/val/test/test_structured.json
  3. duplicate stem/soil on train (same as group-A fpfix)

Output: data/crops3d/all_crops_full_fpfix_attrfunc/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crop_label_text_attrfunc import batch_save_all_crops_attrfunc
from tools.convert_jsonl_to_crops3d_json import convert_entries, load_jsonl, save_json


def apply_fpfix_duplication(train_entries: list[dict], extra_organs: set[str]) -> list[dict]:
    out: list[dict] = []
    for q in train_entries:
        out.append(q)
        if q["object_name"] in extra_organs:
            out.append(dict(q))
    return out


def main():
    p = argparse.ArgumentParser(description="Build all_crops_full_fpfix_attrfunc labels")
    p.add_argument(
        "--crops_root",
        type=str,
        default="Crops3D/Crops3D",
        help="PLY root directory",
    )
    p.add_argument(
        "--jsonl_dir",
        type=str,
        default="data/text_queries_attrfunc",
        help="intermediate JSONL output dir",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/crops3d/all_crops_full_fpfix_attrfunc",
        help="final crops3d JSON output dir",
    )
    p.add_argument(
        "--extra_organs",
        type=str,
        nargs="+",
        default=["stem", "soil"],
        help="organs to duplicate on train (aligned with group-A fpfix)",
    )
    p.add_argument(
        "--skip_jsonl",
        action="store_true",
        help="skip JSONL generation if data/text_queries_attrfunc/ already exists",
    )
    args = p.parse_args()

    crops_root = ROOT / args.crops_root
    jsonl_dir = ROOT / args.jsonl_dir
    output_dir = ROOT / args.output_dir
    split_dir = ROOT / "data" / "crops3d" / "splits"

    if not args.skip_jsonl:
        print("=== Step 1: generate attribute/functional JSONL ===")
        batch_save_all_crops_attrfunc(
            crops_root=crops_root,
            output_root=jsonl_dir,
            project_root=ROOT,
            split_dir=split_dir,
        )
    else:
        print("=== Step 1: skip JSONL (--skip_jsonl) ===")

    print("\n=== Step 2: JSONL → crops3d JSON ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_pairs = [
        ("train_queries.jsonl", "crops3d_train.json"),
        ("val_queries.jsonl", "crops3d_val.json"),
        ("test_queries.jsonl", "crops3d_test.json"),
        ("test_structured_queries.jsonl", "crops3d_test_structured.json"),
    ]
    counts: dict[str, int] = {}
    for jsonl_name, json_name in split_pairs:
        jsonl_path = jsonl_dir / jsonl_name
        if not jsonl_path.exists():
            raise FileNotFoundError(f"missing {jsonl_path}; drop --skip_jsonl and regenerate")
        entries = convert_entries(load_jsonl(jsonl_path))
        save_json(entries, output_dir / json_name)
        counts[json_name] = len(entries)

    print("\n=== Step 3: fpfix stem/soil duplication (train only) ===")
    extra = set(args.extra_organs)
    train_path = output_dir / "crops3d_train.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    before_n = len(train)
    before_vc = Counter(e.get("text_variant", "?") for e in train)
    train_fpfix = apply_fpfix_duplication(train, extra)
    train_path.write_text(json.dumps(train_fpfix, ensure_ascii=False, indent=2), encoding="utf-8")
    after_vc = Counter(e.get("text_variant", "?") for e in train_fpfix)

    summary = {
        "note": "control E: canonical+attribute+functional; train stem/soil x2; same clouds as A",
        "text_protocol": "canonical + attribute + functional",
        "baseline_reference": "exps/crops3d_full_baseline_fpfix_v15/20260720_182330",
        "jsonl_dir": str(jsonl_dir.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "extra_organs": sorted(extra),
        "counts_before_fpfix": counts,
        "train_n_before_fpfix": before_n,
        "train_n_after_fpfix": len(train_fpfix),
        "train_variants_before_fpfix": dict(before_vc),
        "train_variants_after_fpfix": dict(after_vc),
    }
    summary_path = output_dir / "dataset_summary_attrfunc.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDone: {output_dir}/")
    print("Next: bash scripts/train_crops3d_full_fpfix_v15_attrfunc.sh")


if __name__ == "__main__":
    main()
