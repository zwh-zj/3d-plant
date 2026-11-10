"""Build structured test JSONL from canonical test JSONL without rescanning PLY."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crop_label_text import expand_structured_paraphrases, write_jsonl


def load_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate test_structured_queries.jsonl")
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
        help="JSONL directory",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    jsonl_dir = project_root / args.jsonl_dir
    canonical_path = jsonl_dir / "test_queries.jsonl"
    structured_path = jsonl_dir / "test_structured_queries.jsonl"

    if not canonical_path.exists():
        raise FileNotFoundError(
            f"missing {canonical_path}; run: python batch_all_crops.py"
        )

    canonical_entries = load_jsonl(canonical_path)
    structured_entries = expand_structured_paraphrases(canonical_entries)
    write_jsonl(structured_path, structured_entries)

    print(f"canonical test queries : {len(canonical_entries)}")
    print(f"structured test queries: {len(structured_entries)}")
    print(f"Output -> {structured_path}")
    print("Next: python tools/convert_jsonl_to_crops3d_json.py")


if __name__ == "__main__":
    main()
