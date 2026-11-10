#!/usr/bin/env python3
"""Copy crops3d JSON with crop-class names stripped from the text.

Does not modify the source dir; only description changes (scene_id / crop / object_name / GT stay).
Used as a control vs queries that include crop names.

Example:
  python tools/strip_crop_name_from_queries.py \\
    --src data/crops3d/all_crops_full_fpfix \\
    --dst data/crops3d/all_crops_full_fpfix_nocrop
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JSON_NAMES = (
    "crops3d_train.json",
    "crops3d_val.json",
    "crops3d_test.json",
    "crops3d_test_structured.json",
)

# Cotton organ names contain "cotton"; protect them when stripping crop names
_COTTON_ORGAN_PROTECT = [
    (re.compile(r"\bcotton balls\b", re.I), "__COTTON_BALLS__"),
    (re.compile(r"\bcotton ball\b", re.I), "__COTTON_BALL__"),
]


def strip_crop_from_text(text: str, crop: str) -> str:
    """Strip crop-class words; keep organ semantics (including cotton ball)."""
    if not text or not crop:
        return text

    crop_l = crop.lower().strip()
    out = text

    # 1) protect cotton-ball organ phrase (do not delete organ-name cotton)
    if crop_l == "cotton":
        for pat, token in _COTTON_ORGAN_PROTECT:
            out = pat.sub(token, out)

    # 2) high-frequency collocations (longest first)
    replacements = [
        (rf"\bof the {crop_l} crop\b", "of the crop"),
        (rf"\bof the {crop_l} plant\b", "of the plant"),
        (rf"\bin the {crop_l} scene\b", "in the scene"),
        (rf"\bof the {crop_l} scene\b", "of the scene"),
        (rf"\bthe {crop_l} plant organs\b", "the plant organs"),
        (rf"\bthe {crop_l} plant base\b", "the plant base"),
        (rf"\bthe {crop_l} plant\b", "the plant"),
        (rf"\bthe {crop_l} scene\b", "the scene"),
        (rf"\bthe {crop_l} crop\b", "the crop"),
        (
            rf"\b{crop_l} (stem regions|leaf regions|shoot systems|shoot system|"
            rf"stems|leaves|stem|leaf|branches|branch|root region|root)\b",
            r"\1",
        ),
        # leftover standalone crop words (adj/noun)
        (rf"\b{crop_l}\b", ""),
    ]
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.I)

    # 3) restore protected phrases
    if crop_l == "cotton":
        out = out.replace("__COTTON_BALLS__", "cotton balls")
        out = out.replace("__COTTON_BALL__", "cotton ball")

    # 4) whitespace and article cleanup
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"\bthe\s+the\b", "the", out, flags=re.I)
    out = out.strip(" ,.;:")
    # capitalize sentence start
    if out:
        out = out[0].upper() + out[1:]
    return out


def process_entries(entries: list) -> tuple[list, dict]:
    crop_pat = re.compile(
        r"\b(cabbage|cotton|maize|potato|rapeseed|rice|tomato|wheat)\b",
        re.I,
    )
    out = []
    changed = 0
    still_has = 0
    examples = []
    for e in entries:
        ne = dict(e)
        old = e.get("description", "")
        new = strip_crop_from_text(old, e.get("crop", ""))
        ne["description"] = new
        if new != old:
            changed += 1
            if len(examples) < 12:
                examples.append({"crop": e.get("crop"), "before": old, "after": new})
        # cotton inside cotton ball is not a leftover crop class
        check = new
        if e.get("crop", "").lower() == "cotton":
            check = re.sub(r"\bcotton balls?\b", "", check, flags=re.I)
        if crop_pat.search(check):
            still_has += 1
        out.append(ne)
    stats = {
        "n": len(entries),
        "changed": changed,
        "still_has_crop_token": still_has,
        "examples": examples,
    }
    return out, stats


def main():
    p = argparse.ArgumentParser(description="Write crops3d JSON copies with crop-class names stripped")
    p.add_argument("--src", type=str, required=True, help="source annotation dir")
    p.add_argument("--dst", type=str, required=True, help="output dir (must differ from source)")
    p.add_argument(
        "--also-copy-summary",
        action="store_true",
        help="also copy sidecar files such as annotation_summary",
    )
    args = p.parse_args()

    src = ROOT / args.src
    dst = ROOT / args.dst
    if src.resolve() == dst.resolve():
        raise SystemExit("error: --dst must not equal --src (no in-place rewrite)")
    if not src.is_dir():
        raise SystemExit(f"error: source dir missing {src}")

    dst.mkdir(parents=True, exist_ok=True)
    all_stats = {
        "note": "description crop-class stripped; GT/scene/crop unchanged; source not modified",
        "src": str(src.relative_to(ROOT)),
        "dst": str(dst.relative_to(ROOT)),
        "files": {},
    }

    for name in JSON_NAMES:
        sp = src / name
        if not sp.exists():
            print(f"skip missing: {sp}")
            continue
        entries = json.loads(sp.read_text(encoding="utf-8"))
        new_entries, stats = process_entries(entries)
        (dst / name).write_text(
            json.dumps(new_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_stats["files"][name] = {
            "n": stats["n"],
            "changed": stats["changed"],
            "still_has_crop_token": stats["still_has_crop_token"],
            "examples": stats["examples"],
        }
        print(
            f"{name}: n={stats['n']} changed={stats['changed']} "
            f"still_has_crop_token={stats['still_has_crop_token']}"
        )

    # optional: copy sidecar stats (JSON remains the annotation source)
    if args.also_copy_summary:
        for extra in ("annotation_summary.json", "dataset_summary_fpfix.json", "query_count_by_split.json"):
            ep = src / extra
            if ep.exists():
                shutil.copy2(ep, dst / extra)

    summary_path = dst / "nocrop_strip_summary.json"
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {dst}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
