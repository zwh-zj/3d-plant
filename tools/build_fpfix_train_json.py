#!/usr/bin/env python3
"""Duplicate stem/soil on single_maize_full train into single_maize_full_fpfix.

val/test match the source; only train organ exposure changes (stem/soil 8->16).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default="data/crops3d/single_maize_full")
    p.add_argument("--dst", type=str, default="data/crops3d/single_maize_full_fpfix")
    p.add_argument(
        "--extra_organs",
        type=str,
        nargs="+",
        default=["stem", "soil"],
        help="organs to duplicate on existing train items",
    )
    args = p.parse_args()
    src = ROOT / args.src
    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("crops3d_val.json", "crops3d_test.json", "crops3d_test_structured.json"):
        shutil.copy2(src / name, dst / name)

    train = json.loads((src / "crops3d_train.json").read_text(encoding="utf-8"))
    before = Counter(q["object_name"] for q in train)
    extra = set(args.extra_organs)
    out = []
    for q in train:
        out.append(q)
        if q["object_name"] in extra:
            out.append(dict(q))
    after = Counter(q["object_name"] for q in out)
    (dst / "crops3d_train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "note": "duplicate listed organs on baseline train; val/test match src",
        "extra_organs": sorted(extra),
        "train_n_before": len(train),
        "train_n_after": len(out),
        "train_before": dict(before),
        "train_after": dict(after),
    }
    (dst / "dataset_summary_fpfix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
