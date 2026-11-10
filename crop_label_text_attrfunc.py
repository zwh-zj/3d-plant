"""Attribute / functional text templates (control group E; parallel to crop_label_text).

Template style (aligned with the paper figure):
  - canonical : Segment all {region} of the {crop} crop
  - attribute : Locate the {attribute} {object}
  - functional: Segment the {object}, {functional phrase}

Use: contrast with group A (canonical + spatial + relation) to show why
spatial/relation descriptions are preferred over attribute/functional ones.
"""

from __future__ import annotations

from crop_label_text import (
    CROP_CONFIG,
    CROP_NAMES,
    SCENE_GRAPH_STYLE_CONTEXT,
    build_base_queries_for_ply,
    build_canonical_crop_text,
    build_default_scene_graph_style_context,
    get_crop_config,
    get_train_oversample_factor,
    load_labels_from_ply,
    load_split_map,
    to_relative_path,
    write_jsonl,
)


def build_attribute_functional_paraphrases(crop: str, organ_en: str) -> list[str]:
    """Turn SCENE_GRAPH attribute / functional context into NL queries."""
    context = SCENE_GRAPH_STYLE_CONTEXT.get((crop, organ_en)) or build_default_scene_graph_style_context(
        crop, organ_en
    )
    if not context:
        return []

    obj = context["object"]
    texts: list[str] = []
    texts.extend(f"Locate the {attr} {obj}" for attr in context.get("attributes", []))
    texts.extend(f"Segment the {obj}, {phrase}" for phrase in context.get("functional", []))
    return texts


def infer_text_variant(text: str) -> str:
    if text.startswith("Locate the"):
        return "attribute"
    if text.startswith("Segment the") and ", " in text:
        return "functional"
    return "canonical"


def append_attrfunc_variants(entry: dict, crop: str) -> list[dict]:
    variants = [dict(entry)]
    variants[0]["text_variant"] = "canonical"
    for idx, text in enumerate(build_attribute_functional_paraphrases(crop, entry["organ_en"]), start=1):
        variant = dict(entry)
        variant["text"] = text
        variant["sample_id"] = f"{entry['sample_id']}_af{idx}"
        variant["is_paraphrase"] = True
        variant["canonical_sample_id"] = entry["sample_id"]
        variant["text_variant"] = infer_text_variant(text)
        variants.append(variant)
    return variants


def expand_attrfunc_paraphrases(entries: list[dict], crop: str | None = None) -> list[dict]:
    expanded: list[dict] = []
    for entry in entries:
        entry_crop = crop or entry["crop"]
        expanded.extend(append_attrfunc_variants(entry, entry_crop))
    return expanded


def expand_train_attrfunc_and_oversample(entries: list[dict], crop: str, split: str) -> list[dict]:
    if split not in {"train", "val"}:
        return entries
    expanded: list[dict] = []
    for entry in entries:
        variants = append_attrfunc_variants(entry, crop)
        target_factor = get_train_oversample_factor(crop, entry["organ_en"])
        if split == "train":
            extra_needed = max(0, target_factor - len(variants))
        else:
            extra_needed = 0
        for idx in range(extra_needed):
            source = variants[idx % len(variants)]
            duplicate = dict(source)
            duplicate["sample_id"] = f"{entry['sample_id']}_os{idx + 1}"
            duplicate["is_oversampled"] = True
            duplicate["canonical_sample_id"] = entry["sample_id"]
            variants.append(duplicate)
        expanded.extend(variants)
    return expanded


def batch_save_text_queries_attrfunc(
    input_dir,
    output_dir,
    label_map,
    exclude_labels,
    crop,
    project_root,
    split_dir,
    composite_organs=None,
):
    """One crop; train/val use attribute+functional; test stays canonical + structured test."""
    from pathlib import Path

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = load_split_map(split_dir, crop=crop)
    ply_files = sorted(input_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {input_dir}")

    all_entries = []
    train_entries = []
    val_entries = []
    test_entries = []
    test_structured_entries = []
    errors = []

    total = len(ply_files)
    for i, ply_path in enumerate(ply_files, start=1):
        try:
            split = split_map.get(ply_path.name, "unknown")
            labels = load_labels_from_ply(ply_path)
            base_entries = build_base_queries_for_ply(
                ply_path=ply_path,
                labels=labels,
                crop=crop,
                split=split,
                project_root=project_root,
                label_map=label_map,
                exclude_labels=exclude_labels,
                composite_organs=composite_organs,
            )
            if split == "test":
                entries = base_entries
                structured_entries = expand_attrfunc_paraphrases(base_entries, crop)
                test_entries.extend(entries)
                test_structured_entries.extend(structured_entries)
            else:
                entries = expand_train_attrfunc_and_oversample(base_entries, crop, split)
                structured_entries = []
                if split == "train":
                    train_entries.extend(entries)
                elif split == "val":
                    val_entries.extend(entries)

            all_entries.extend(entries)
            all_entries.extend(structured_entries)

            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}] {ply_path.name} ({split})")
        except Exception as exc:
            errors.append({"file": ply_path.name, "error": str(exc)})
            print(f"  [{i}/{total}] FAILED {ply_path.name}: {exc}")

    paths = {
        "all": output_dir / "all_queries.jsonl",
        "train": output_dir / "train_queries.jsonl",
        "val": output_dir / "val_queries.jsonl",
        "test": output_dir / "test_queries.jsonl",
        "test_structured": output_dir / "test_structured_queries.jsonl",
    }
    write_jsonl(paths["all"], all_entries)
    write_jsonl(paths["train"], train_entries)
    write_jsonl(paths["val"], val_entries)
    write_jsonl(paths["test"], test_entries)
    write_jsonl(paths["test_structured"], test_structured_entries)

    import json

    index = {
        "crop": crop,
        "language": "en",
        "task": "organ_segmentation",
        "text_protocol": "canonical+attribute+functional",
        "input_dir": to_relative_path(input_dir, project_root),
        "output_dir": to_relative_path(output_dir, project_root),
        "excluded_labels": sorted(exclude_labels),
        "num_point_clouds": total - len(errors),
        "num_queries": len(all_entries),
        "num_train_queries": len(train_entries),
        "num_val_queries": len(val_entries),
        "num_test_queries": len(test_entries),
        "num_test_structured_queries": len(test_structured_entries),
        "num_errors": len(errors),
        "errors": errors,
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return {
        "all_entries": all_entries,
        "train_entries": train_entries,
        "val_entries": val_entries,
        "test_entries": test_entries,
        "test_structured_entries": test_structured_entries,
        **{k: index[k] for k in (
            "num_point_clouds", "num_queries", "num_train_queries",
            "num_val_queries", "num_test_queries", "num_test_structured_queries",
            "num_errors",
        )},
    }


def batch_save_all_crops_attrfunc(crops_root, output_root, project_root, split_dir):
    """Attribute/functional text JSONL for all 8 crops."""
    from collections import Counter
    from pathlib import Path
    import json

    crops_root = Path(crops_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_all = []
    global_train = []
    global_val = []
    global_test = []
    global_test_structured = []
    summary = {
        "project_root": ".",
        "text_protocol": "canonical+attribute+functional",
        "crops_root": to_relative_path(crops_root, project_root),
        "output_root": to_relative_path(output_root, project_root),
        "split_dir": to_relative_path(split_dir, project_root),
        "crops": {},
        "num_point_clouds": 0,
        "num_queries": 0,
        "num_train_queries": 0,
        "num_val_queries": 0,
        "num_test_queries": 0,
        "num_test_structured_queries": 0,
        "num_errors": 0,
    }

    for crop in CROP_NAMES:
        cfg = get_crop_config(crop)
        input_dir = crops_root / crop
        output_dir = output_root / crop

        print(f"\n=== {crop} ({len(list(input_dir.glob('*.ply')))} PLY) ===")
        result = batch_save_text_queries_attrfunc(
            input_dir=input_dir,
            output_dir=output_dir,
            label_map=cfg["label_map"],
            exclude_labels=cfg["exclude_labels"],
            crop=crop,
            project_root=project_root,
            split_dir=split_dir,
            composite_organs=cfg.get("composite_organs", []),
        )

        global_all.extend(result["all_entries"])
        global_train.extend(result["train_entries"])
        global_val.extend(result["val_entries"])
        global_test.extend(result["test_entries"])
        global_test_structured.extend(result["test_structured_entries"])

        train_vc = Counter(e.get("text_variant", "?") for e in result["train_entries"])
        summary["crops"][crop] = {
            "num_point_clouds": result["num_point_clouds"],
            "num_queries": result["num_queries"],
            "num_train_queries": result["num_train_queries"],
            "num_val_queries": result["num_val_queries"],
            "num_test_queries": result["num_test_queries"],
            "num_test_structured_queries": result["num_test_structured_queries"],
            "num_errors": result["num_errors"],
            "train_variants": dict(train_vc),
            "excluded_labels": sorted(cfg["exclude_labels"]),
        }
        for key in (
            "num_point_clouds", "num_queries", "num_train_queries",
            "num_val_queries", "num_test_queries", "num_test_structured_queries", "num_errors",
        ):
            summary[key] += result[key]

        print(
            f"  -> queries={result['num_queries']} "
            f"(train={result['num_train_queries']}, val={result['num_val_queries']}, "
            f"test={result['num_test_queries']}, test_structured={result['num_test_structured_queries']})"
        )

    write_jsonl(output_root / "all_queries.jsonl", global_all)
    write_jsonl(output_root / "train_queries.jsonl", global_train)
    write_jsonl(output_root / "val_queries.jsonl", global_val)
    write_jsonl(output_root / "test_queries.jsonl", global_test)
    write_jsonl(output_root / "test_structured_queries.jsonl", global_test_structured)

    summary["train_variants"] = dict(Counter(e.get("text_variant", "?") for e in global_train))
    summary_path = output_root / "index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "summary_path": summary_path,
        "summary": summary,
    }
