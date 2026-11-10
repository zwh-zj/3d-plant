#!/usr/bin/env python3
"""Build group-E attribute/functional labels from group-A fpfix JSON (preferred path).

Does not read PLY or modify all_crops_full_fpfix/; only description / text_variant change.
Keeps scene_id / object_ids / split and other geometry/split fields.

Usage:
  python tools/build_attrfunc_fpfix_from_json.py
  python tools/build_attrfunc_fpfix_from_json.py \\
    --src data/crops3d/all_crops_full_fpfix \\
    --dst data/crops3d/all_crops_full_fpfix_attrfunc
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crop_label_text_attrfunc import (
    append_attrfunc_variants,
    build_attribute_functional_paraphrases,
    infer_text_variant,
)


SPLIT_FILES = (
    "crops3d_train.json",
    "crops3d_val.json",
    "crops3d_test.json",
    "crops3d_test_structured.json",
)


def _canonical_key(entry: dict) -> str:
    return entry.get("canonical_sample_id") or entry.get("sample_id") or (
        f"{entry['scene_id']}|{entry['ann_id']}|{entry['object_name']}"
    )


def _entry_to_base(entry: dict) -> dict:
    """Rebuild attrfunc base fields from an MDIN JSON item."""
    return {
        "sample_id": entry.get("sample_id") or f"{entry['scene_id']}__{entry['object_name']}",
        "split": entry.get("split", "unknown"),
        "task": "organ_segmentation",
        "crop": entry["crop"],
        "point_cloud": entry.get("point_cloud", ""),
        "text": entry["description"],
        "organ_en": entry["object_name"],
        "point_count": entry.get("point_count", 0),
        "label_id": entry["object_ids"][0] if len(entry.get("object_ids", [])) == 1 else None,
        "label_ids": entry["object_ids"] if len(entry.get("object_ids", [])) > 1 else None,
    }


def _variant_to_mdin(base_entry: dict, variant: dict, template: dict) -> dict:
    out = dict(template)
    out["description"] = variant["text"]
    out["text_variant"] = variant.get("text_variant", infer_text_variant(variant["text"]))
    out["sample_id"] = variant.get("sample_id", out.get("sample_id", ""))
    if variant.get("is_paraphrase"):
        out["canonical_sample_id"] = variant.get("canonical_sample_id", _canonical_key(base_entry))
    if variant.get("is_oversampled"):
        out["is_oversampled"] = True
    return out


def transform_split_entries(entries: list[dict], split_name: str) -> tuple[list[dict], dict]:
    if split_name in {"crops3d_test.json", "crops3d_test_structured.json"}:
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            if e.get("text_variant", "canonical") == "canonical":
                key = e.get("sample_id") or f"{e['scene_id']}|{e['ann_id']}"
                groups[key].append(e)
            else:
                groups[_canonical_key(e)].append(e)

        out: list[dict] = []
        for key, group in groups.items():
            canonical = next((g for g in group if g.get("text_variant", "canonical") == "canonical"), group[0])
            base = _entry_to_base(canonical)
            variants = append_attrfunc_variants(base, canonical["crop"])
            if split_name == "crops3d_test.json":
                variants = [variants[0]]
            template = dict(canonical)
            for v in variants:
                out.append(_variant_to_mdin(base, v, template))
        stats = {
            "input_n": len(entries),
            "output_n": len(out),
            "output_by_variant": dict(Counter(e.get("text_variant", "?") for e in out)),
        }
        return out, stats

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[_canonical_key(e)].append(e)

    out = []
    for _, group in groups.items():
        canonical = next((g for g in group if g.get("text_variant", "canonical") == "canonical"), group[0])
        base = _entry_to_base(canonical)
        base_variants = append_attrfunc_variants(base, canonical["crop"])
        template = dict(canonical)

        non_os = [g for g in group if not g.get("is_oversampled")]
        os_entries = [g for g in group if g.get("is_oversampled")]

        for i, src in enumerate(non_os):
            variant = base_variants[i] if i < len(base_variants) else base_variants[i % len(base_variants)]
            out.append(_variant_to_mdin(base, variant, dict(src, description=variant["text"], text_variant=variant["text_variant"])))

        for j, src in enumerate(os_entries):
            src_variant = src.get("text_variant", "canonical")
            pool = [v for v in base_variants if v.get("text_variant") == src_variant]
            if not pool:
                pool = base_variants
            pick = pool[j % len(pool)]
            dup = dict(src)
            dup["description"] = pick["text"]
            dup["text_variant"] = pick["text_variant"]
            dup["is_oversampled"] = True
            out.append(dup)

    stats = {
        "input_n": len(entries),
        "output_n": len(out),
        "input_by_variant": dict(Counter(e.get("text_variant", "?") for e in entries)),
        "output_by_variant": dict(Counter(e.get("text_variant", "?") for e in out)),
    }
    return out, stats


def main():
    p = argparse.ArgumentParser(description="Build attribute/functional labels from group-A fpfix JSON")
    p.add_argument("--src", type=str, default="data/crops3d/all_crops_full_fpfix")
    p.add_argument("--dst", type=str, default="data/crops3d/all_crops_full_fpfix_attrfunc")
    args = p.parse_args()

    src = ROOT / args.src
    dst = ROOT / args.dst
    if src.resolve() == dst.resolve():
        raise SystemExit("error: --dst must not equal --src")
    dst.mkdir(parents=True, exist_ok=True)

    summary = {
        "note": "text converted from group-A JSON; geometry/scene_id/object_ids unchanged",
        "src": str(src.relative_to(ROOT)),
        "dst": str(dst.relative_to(ROOT)),
        "text_protocol": "canonical + attribute + functional",
        "files": {},
        "example_templates": {
            "Maize_tassel": {
                "attribute": "Locate the flowering tassels",
                "functional": "Segment the tassels, the male flowering structures of the maize plant",
                "canonical": "Segment all tassel regions of the maize crop",
            }
        },
    }

    for name in SPLIT_FILES:
        sp = src / name
        if not sp.exists():
            print(f"skip missing: {sp}")
            continue
        entries = json.loads(sp.read_text(encoding="utf-8"))
        transformed, stats = transform_split_entries(entries, name)
        (dst / name).write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["files"][name] = stats
        print(f"{name}: {stats['input_n']} -> {stats['output_n']}  {stats.get('output_by_variant', {})}")

    summary_path = dst / "dataset_summary_attrfunc.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone: {dst}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
