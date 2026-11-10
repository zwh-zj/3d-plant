"""Crops3D organ-label to English prompt mapping (paper Table 2).

Step 1 core logic:
- read scalar_sf organ labels from PLY
- map crop type to English queries (e.g. "Segment all leaf regions of the cabbage crop")
- write JSONL for convert / training
"""

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

CANONICAL_TEXT_TEMPLATE = "Segment all {region_label} of the {crop} crop"


def organ_en_to_region_label(organ_en):
    """Map organ_en to region_label, e.g. leaf -> leaf regions."""
    return f"{organ_en.replace('_', ' ')} regions"


def build_canonical_crop_text(crop, organ_en):
    """Build canonical text: Segment all {region_label} of the {crop} crop."""
    return CANONICAL_TEXT_TEMPLATE.format(
        region_label=organ_en_to_region_label(organ_en),
        crop=crop.lower(),
    )


def normalize_label_map_organ(value):
    """Accept label_map values as organ_en or legacy (organ_en, text)."""
    if isinstance(value, tuple):
        return value[0]
    return value


CROP_CONFIG = {
    "Cabbage": {
        "label_map": {
            0: "soil",
            1: "stem",
            2: "leaf",
            3: "pot",
            4: "land",
        },
        "exclude_labels": set(),
        "composite_organs": [
            {
                "organ_en": "shoot_system",
                "label_ids": [1, 2],  # stem + leaf, virtual shoot system
            },
        ],
    },
    "Cotton": {
        "label_map": {
            0: "soil",
            1: "shoot_system",  # treat branch as shoot system
            2: "cotton_ball",
        },
        "exclude_labels": set(),
    },
    "Maize": {
        "label_map": {
            0: "soil",
            1: "stem",
            2: "leaf",
            3: "tassel",
            4: "corn",
            5: "pot",
        },
        "exclude_labels": set(),
        "composite_organs": [
            {
                "organ_en": "shoot_system",
                "label_ids": [1, 2],
            },
        ],
    },
    "Potato": {
        "label_map": {
            0: "stem",
            1: "leaf",
        },
        "exclude_labels": set(),
        "composite_organs": [
            {
                "organ_en": "shoot_system",
                "label_ids": [0, 1],
            },
        ],
    },
    "Rapeseed": {
        "label_map": {
            0: "stem",
            1: "leaf",
            2: "pot",
            3: "soil",
        },
        "exclude_labels": set(),
        "composite_organs": [
            {
                "organ_en": "shoot_system",
                "label_ids": [0, 1],
            },
        ],
    },
    "Rice": {
        "label_map": {
            0: "shoot_system",
            1: "pot",
        },
        "exclude_labels": set(),
    },
    "Tomato": {
        "label_map": {
            0: "stem",
            1: "leaf",
            5: "corn",  # tomato fruit; PLY scalar_sf=5
        },
        # some PLY files have leftover sf=2; skip that id when building queries
        "exclude_labels": {2},
        "composite_organs": [
            {
                "organ_en": "shoot_system",
                "label_ids": [0, 1],  # stem + leaf
            },
        ],
    },
    "Wheat": {
        "label_map": {
            0: "rhizobox",
            1: "shoot_system",
        },
        "exclude_labels": set(),
    },
}

CROP_NAMES = list(CROP_CONFIG.keys())

# Hand-written paraphrase augmentation is disabled; use SCENE_GRAPH_STYLE_CONTEXT only.
TRAIN_PARAPHRASES = {}


SCENE_GRAPH_STYLE_CONTEXT = {
    # Structured semantic context: object / attribute / spatial / relation / functional.
    # Emit natural-language queries only; no graph_node / graph_edge, no model-input change.
    ("Maize", "corn"): {
        "object": "corn ears",
        "attributes": ["ear-like"],
        "spatial": [
            "in the middle part of the maize plant",
            "below the tassel",
        ],
        "relations": [
            "attached to the maize stem",
        ],
        "functional": [
            "the grain-bearing organs of the maize plant",
        ],
    },
    ("Maize", "tassel"): {
        "object": "tassels",
        "attributes": ["flowering"],
        "spatial": [
            "at the top of the maize plant",
            "above the leaves and corn ears",
        ],
        "relations": [
            "connected to the upper shoot system",
        ],
        "functional": [
            "the male flowering structures of the maize plant",
        ],
    },
    ("Tomato", "corn"): {
        "object": "corn-like organs",
        "attributes": ["ear-like"],
        "spatial": [
            "near the tomato stem",
            "among the tomato leaves",
        ],
        "relations": [
            "attached to the tomato shoot system",
        ],
        "functional": [
            "the corn-labeled organs in the tomato plant",
        ],
    },
    ("Wheat", "rhizobox"): {
        "object": "rhizobox regions",
        "attributes": ["rectangular"],
        "spatial": [
            "below the wheat shoot system",
            "around the root observation area",
        ],
        "relations": [
            "enclosing the wheat root region",
        ],
        "functional": [
            "the root observation box around the wheat plant",
        ],
    },
    ("Wheat", "shoot_system"): {
        "object": "wheat shoot systems",
        "attributes": ["above-ground"],
        "spatial": [
            "above the rhizobox",
        ],
        "relations": [
            "consisting of wheat stems and leaves",
        ],
        "functional": [
            "the aerial plant parts of the wheat scene",
        ],
    },
    ("Rice", "shoot_system"): {
        "object": "rice shoot systems",
        "attributes": ["above-ground"],
        "spatial": [
            "above the pot region",
        ],
        "relations": [
            "consisting of rice stems and leaves",
        ],
        "functional": [
            "the visible aerial organs of the rice plant",
        ],
    },
    ("Rice", "pot"): {
        "object": "pot regions",
        "attributes": ["container-like"],
        "spatial": [
            "below the rice shoot system",
        ],
        "relations": [
            "supporting the rice plant base",
        ],
        "functional": [
            "the container holding the rice plant",
        ],
    },
    ("Tomato", "leaf"): {
        "object": "tomato leaf regions",
        "attributes": ["leaf-like"],
        "spatial": [
            "around the tomato stem",
        ],
        "relations": [
            "attached to the tomato shoot system",
        ],
        "functional": [
            "the photosynthetic organs of the tomato plant",
        ],
    },
    ("Tomato", "stem"): {
        "object": "tomato stem regions",
        "attributes": ["supporting"],
        "spatial": [
            "in the central part of the tomato plant",
        ],
        "relations": [
            "supporting tomato leaves and corn-like organs",
        ],
        "functional": [
            "the main structural organs of the tomato plant",
        ],
    },
    ("Cabbage", "land"): {
        "object": "land regions",
        "attributes": ["ground"],
        "spatial": [
            "below the cabbage plant",
        ],
        "relations": [
            "surrounding the plant base",
        ],
        "functional": [
            "the support surface under the cabbage scene",
        ],
    },
    ("Cotton", "cotton_ball"): {
        "object": "cotton balls",
        "attributes": ["boll-like"],
        "spatial": [
            "on the cotton shoot system",
        ],
        "relations": [
            "attached to cotton branches",
        ],
        "functional": [
            "the cotton-producing organs of the plant",
        ],
    },
}


TRAIN_OVERSAMPLE_FACTORS = {
    ("Cabbage", "land"): 8,
    ("Maize", "tassel"): 20,  # 10 scene × 20 = 200 train queries
    ("Maize", "corn"): 25,    # 8 scene × 25 = 200 train queries
}


def get_train_paraphrases(crop, organ_en):
    return TRAIN_PARAPHRASES.get((crop, organ_en), []) + build_scene_graph_style_paraphrases(
        crop, organ_en
    )


def build_default_scene_graph_style_context(crop, organ_en):
    """Conservative context for organs without explicit config, to avoid uncertain relations."""
    crop_name = crop.lower()
    defaults = {
        "soil": {
            "object": "soil regions",
            "attributes": ["ground-like"],
            "spatial": [
                f"below the {crop_name} plant organs",
            ],
            "relations": [
                f"surrounding the {crop_name} plant base",
            ],
            "functional": [
                f"the ground support region in the {crop_name} scene",
            ],
        },
        "stem": {
            "object": f"{crop_name} stem regions",
            "attributes": ["supporting"],
            "spatial": [
                f"in the central structural part of the {crop_name} plant",
            ],
            "relations": [
                f"connected to {crop_name} leaves",
            ],
            "functional": [
                f"the main support organs of the {crop_name} plant",
            ],
        },
        "leaf": {
            "object": f"{crop_name} leaf regions",
            "attributes": ["leaf-like"],
            "spatial": [
                f"around the {crop_name} stem regions",
            ],
            "relations": [
                f"connected to the {crop_name} shoot system",
            ],
            "functional": [
                f"the photosynthetic organs of the {crop_name} plant",
            ],
        },
        "pot": {
            "object": "pot regions",
            "attributes": ["container-like"],
            "spatial": [
                f"below the {crop_name} plant organs",
            ],
            "relations": [
                f"supporting the {crop_name} plant base",
            ],
            "functional": [
                f"the container region holding the {crop_name} plant",
            ],
        },
        "shoot_system": {
            "object": f"{crop_name} shoot systems",
            "attributes": ["above-ground"],
            "spatial": [
                f"above the lower support region of the {crop_name} scene",
            ],
            "relations": [
                f"consisting of visible {crop_name} stems and leaves",
            ],
            "functional": [
                f"the visible aerial plant parts of the {crop_name} scene",
            ],
        },
    }
    return defaults.get(organ_en)


def build_scene_graph_style_paraphrases(crop, organ_en):
    """Turn spatial/relation context into NL queries for train/val text augmentation."""
    context = SCENE_GRAPH_STYLE_CONTEXT.get((crop, organ_en)) or build_default_scene_graph_style_context(
        crop, organ_en
    )
    if not context:
        return []

    obj = context["object"]
    texts = []
    # Keep spatial / relation only, to cut query count and long-run load.
    texts.extend(f"Locate the {obj} {spatial}" for spatial in context.get("spatial", []))
    texts.extend(f"Segment the {obj} {relation}" for relation in context.get("relations", []))
    return texts


def get_train_oversample_factor(crop, organ_en):
    """Train-set oversample count; default 1 if unset."""
    return max(1, int(TRAIN_OVERSAMPLE_FACTORS.get((crop, organ_en), 1)))


def infer_text_variant(text):
    """Infer text type from the description for grouped test stats."""
    if text.startswith("Locate the"):
        return "spatial"
    if text.startswith("Segment the"):
        return "relation"
    return "canonical"


def append_paraphrase_variants(entry, crop):
    """Append spatial/relation structured variants to one canonical query."""
    variants = [dict(entry)]
    variants[0]["text_variant"] = "canonical"
    for idx, text in enumerate(get_train_paraphrases(crop, entry["organ_en"]), start=1):
        variant = dict(entry)
        variant["text"] = text
        variant["sample_id"] = f"{entry['sample_id']}_p{idx}"
        variant["is_paraphrase"] = True
        variant["canonical_sample_id"] = entry["sample_id"]
        variant["text_variant"] = infer_text_variant(text)
        variants.append(variant)
    return variants


def expand_structured_paraphrases(entries, crop=None):
    """Append spatial/relation text without oversampling; usable as a standalone test set."""
    expanded = []
    for entry in entries:
        entry_crop = crop or entry["crop"]
        expanded.extend(append_paraphrase_variants(entry, entry_crop))
    return expanded


def expand_train_paraphrases_and_oversample(entries, crop, split):
    """Add structured queries and oversampling on train/val; keep test canonical."""
    if split not in {"train", "val"}:
        return entries
    expanded = []
    for entry in entries:
        variants = append_paraphrase_variants(entry, crop)

        target_factor = get_train_oversample_factor(crop, entry["organ_en"])
        # Oversample copies on train only; val keeps paraphrases without _os copies
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


def get_crop_config(crop):
    if crop not in CROP_CONFIG:
        raise KeyError(f"Unknown crop: {crop}. Available: {CROP_NAMES}")
    return CROP_CONFIG[crop]


def load_split_map(split_dir, crop=None):
    """
    Load train / val / test splits keyed by PLY filename.
    Prefer the three split files under data/crops3d/splits/.
    Also index *_10k.ply without the _10k suffix so full clouds match.
    """
    split_dir = Path(split_dir)
    split_map = {}

    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"{split_name}.txt"
        if not split_file.exists():
            continue
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip().replace("\\", "/")
            if not line:
                continue
            if crop and f"/{crop}/" not in line:
                continue
            name = Path(line).name
            split_map[name] = split_name
            if name.endswith("_10k.ply"):
                split_map[name[: -len("_10k.ply")] + ".ply"] = split_name

    return split_map


def to_relative_path(path, project_root):
    """Path relative to the project root (Windows backslash style)."""
    rel = Path(path).resolve().relative_to(Path(project_root).resolve())
    return str(rel)


def load_labels_from_ply(ply_path):
    ply = PlyData.read(ply_path)
    return np.array(ply["vertex"]["scalar_sf"], dtype=np.int64)


def build_base_queries_for_ply(
    ply_path,
    labels,
    crop,
    split,
    project_root,
    label_map,
    exclude_labels,
    composite_organs=None,
):
    """Build canonical JSONL queries for one PLY (one query per present label_id)."""
    composite_organs = composite_organs or []
    ply_path = Path(ply_path)
    point_cloud = to_relative_path(ply_path, project_root)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    present = set(count_map.keys())

    entries = []
    for label_id, count in zip(unique.tolist(), counts.tolist()):
        if label_id in exclude_labels:
            continue

        if label_id in label_map:
            organ_en = normalize_label_map_organ(label_map[label_id])
            text = build_canonical_crop_text(crop, organ_en)
        else:
            organ_en = "unknown"
            text = f"Segment all regions with label {label_id} of the {crop.lower()} crop"

        entries.append(
            {
                "sample_id": f"{crop}/{ply_path.stem}__{organ_en}",
                "split": split,
                "task": "organ_segmentation",
                "crop": crop,
                "point_cloud": point_cloud,
                "text": text,
                "label_id": int(label_id),
                "organ_en": organ_en,
                "point_count": int(count),
            }
        )

    for comp in composite_organs:
        label_ids = [int(x) for x in comp["label_ids"]]
        if not any(lid in present for lid in label_ids):
            continue
        organ_en = comp["organ_en"]
        text = build_canonical_crop_text(crop, organ_en)
        point_count = sum(count_map.get(lid, 0) for lid in label_ids if lid in present)
        entries.append(
            {
                "sample_id": f"{crop}/{ply_path.stem}__{organ_en}",
                "split": split,
                "task": "organ_segmentation",
                "crop": crop,
                "point_cloud": point_cloud,
                "text": text,
                "label_ids": label_ids,
                "organ_en": organ_en,
                "point_count": int(point_count),
            }
        )

    return entries


def build_queries_for_ply(
    ply_path,
    labels,
    crop,
    split,
    project_root,
    label_map,
    exclude_labels,
    composite_organs=None,
):
    """JSONL queries for one PLY; structured augment on train/val, canonical on test."""
    entries = build_base_queries_for_ply(
        ply_path=ply_path,
        labels=labels,
        crop=crop,
        split=split,
        project_root=project_root,
        label_map=label_map,
        exclude_labels=exclude_labels,
        composite_organs=composite_organs,
    )
    return expand_train_paraphrases_and_oversample(entries, crop, split)


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def batch_save_text_queries(
    input_dir,
    output_dir,
    label_map,
    exclude_labels,
    crop,
    project_root,
    split_dir,
    composite_organs=None,
):
    """Process one crop dir; write combined JSONL and index.json."""
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
                structured_entries = expand_structured_paraphrases(base_entries, crop)
                test_entries.extend(entries)
                test_structured_entries.extend(structured_entries)
            else:
                entries = expand_train_paraphrases_and_oversample(base_entries, crop, split)
                structured_entries = []
                if split == "train":
                    train_entries.extend(entries)
                elif split == "val":
                    val_entries.extend(entries)

            all_entries.extend(entries)
            all_entries.extend(structured_entries)

            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}] {ply_path.name} ({split})")
        except Exception as e:
            errors.append({"file": ply_path.name, "error": str(e)})
            print(f"  [{i}/{total}] FAILED {ply_path.name}: {e}")

    all_jsonl_path = output_dir / "all_queries.jsonl"
    train_jsonl_path = output_dir / "train_queries.jsonl"
    val_jsonl_path = output_dir / "val_queries.jsonl"
    test_jsonl_path = output_dir / "test_queries.jsonl"
    test_structured_jsonl_path = output_dir / "test_structured_queries.jsonl"

    write_jsonl(all_jsonl_path, all_entries)
    write_jsonl(train_jsonl_path, train_entries)
    write_jsonl(val_jsonl_path, val_entries)
    write_jsonl(test_jsonl_path, test_entries)
    write_jsonl(test_structured_jsonl_path, test_structured_entries)

    index = {
        "crop": crop,
        "language": "en",
        "task": "organ_segmentation",
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
        "all_jsonl_path": all_jsonl_path,
        "train_jsonl_path": train_jsonl_path,
        "val_jsonl_path": val_jsonl_path,
        "test_jsonl_path": test_jsonl_path,
        "test_structured_jsonl_path": test_structured_jsonl_path,
        "index_path": index_path,
        **{k: index[k] for k in (
            "num_point_clouds", "num_queries", "num_train_queries",
            "num_val_queries", "num_test_queries", "num_test_structured_queries",
            "num_errors",
        )},
    }


def batch_save_all_crops(crops_root, output_root, project_root, split_dir):
    """Process all 8 crops; write train / val / test / test_structured JSONL."""
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
        result = batch_save_text_queries(
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

        summary["crops"][crop] = {
            "num_point_clouds": result["num_point_clouds"],
            "num_queries": result["num_queries"],
            "num_train_queries": result["num_train_queries"],
            "num_val_queries": result["num_val_queries"],
            "num_test_queries": result["num_test_queries"],
            "num_test_structured_queries": result["num_test_structured_queries"],
            "num_errors": result["num_errors"],
            "excluded_labels": sorted(cfg["exclude_labels"]),
        }
        summary["num_point_clouds"] += result["num_point_clouds"]
        summary["num_queries"] += result["num_queries"]
        summary["num_train_queries"] += result["num_train_queries"]
        summary["num_val_queries"] += result["num_val_queries"]
        summary["num_test_queries"] += result["num_test_queries"]
        summary["num_test_structured_queries"] += result["num_test_structured_queries"]
        summary["num_errors"] += result["num_errors"]

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

    summary_path = output_root / "index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "summary_path": summary_path,
        "all_jsonl_path": output_root / "all_queries.jsonl",
        "train_jsonl_path": output_root / "train_queries.jsonl",
        "val_jsonl_path": output_root / "val_queries.jsonl",
        "test_jsonl_path": output_root / "test_queries.jsonl",
        "test_structured_jsonl_path": output_root / "test_structured_queries.jsonl",
        "summary": summary,
    }
