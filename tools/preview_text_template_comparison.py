#!/usr/bin/env python3
"""Side-by-side preview of group A (spatial+relation) vs E (attribute+functional).

Does not modify labels; only samples crop_label_text / crop_label_text_attrfunc.

Usage:
  python tools/preview_text_template_comparison.py
  python tools/preview_text_template_comparison.py --crop Maize --organ tassel
  python tools/preview_text_template_comparison.py --markdown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crop_label_text import (
    CROP_CONFIG,
    build_canonical_crop_text,
    build_scene_graph_style_paraphrases,
)
from crop_label_text_attrfunc import build_attribute_functional_paraphrases


def collect_organs(crop: str) -> list[str]:
    cfg = CROP_CONFIG[crop]
    organs = {v if isinstance(v, str) else v[0] for v in cfg["label_map"].values()}
    for comp in cfg.get("composite_organs", []):
        organs.add(comp["organ_en"])
    return sorted(organs)


def build_rows(crop: str, organ_en: str) -> list[dict]:
    canonical = build_canonical_crop_text(crop, organ_en)
    spatial_relation = build_scene_graph_style_paraphrases(crop, organ_en)
    attr_func = build_attribute_functional_paraphrases(crop, organ_en)

    rows = [{"group": "A / E shared", "variant": "canonical", "template": "Segment all {region} of the {crop} crop", "example": canonical}]
    for i, text in enumerate(spatial_relation, 1):
        kind = "spatial" if text.startswith("Locate the") else "relation"
        tpl = (
            "Locate the {object} {spatial phrase}"
            if kind == "spatial"
            else "Segment the {object} {relation phrase}"
        )
        rows.append({"group": "A (current)", "variant": kind, "template": tpl, "example": text})
    for i, text in enumerate(attr_func, 1):
        kind = "attribute" if text.startswith("Locate the") else "functional"
        tpl = (
            "Locate the {attribute} {object}"
            if kind == "attribute"
            else "Segment the {object}, {functional phrase}"
        )
        rows.append({"group": "E (control)", "variant": kind, "template": tpl, "example": text})
    return rows


def print_table(rows: list[dict]):
    print(f"{'group':<12} {'variant':<12} {'example':<72} template")
    print("-" * 130)
    for r in rows:
        print(f"{r['group']:<10} {r['variant']:<12} {r['example']:<72} <- {r['template']}")


def print_markdown(rows: list[dict], crop: str, organ_en: str):
    print(f"### {crop} · {organ_en}\n")
    print("| group | variant | example | template |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        ex = r["example"].replace("|", "\\|")
        print(f"| {r['group']} | {r['variant']} | `{ex}` | {r['template']} |")
    print()


def main():
    p = argparse.ArgumentParser(description="Preview group A vs E text templates")
    p.add_argument("--crop", type=str, default=None, help="crop name, e.g. Maize")
    p.add_argument("--organ", type=str, default=None, help="organ name, e.g. tassel")
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = p.parse_args()

    crops = [args.crop] if args.crop else list(CROP_CONFIG.keys())
    for crop in crops:
        if crop not in CROP_CONFIG:
            raise SystemExit(f"unknown crop: {crop}")
        organs = [args.organ] if args.organ else collect_organs(crop)
        for organ_en in organs:
            rows = build_rows(crop, organ_en)
            if args.markdown:
                print_markdown(rows, crop, organ_en)
            else:
                print(f"\n{'=' * 130}")
                print(f"{crop} · {organ_en}")
                print_table(rows)


if __name__ == "__main__":
    main()
