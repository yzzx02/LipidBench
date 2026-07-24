from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.data import (
    BASE_ATTRIBUTE_NAMES,
    load_manifest_jsonl,
    save_manifest_jsonl,
    split_grouped_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe grouped PeakMultiTaskRCNN manifests.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--attr-dim", type=int, default=len(BASE_ATTRIBUTE_NAMES))
    parser.add_argument(
        "--group-by",
        choices=("source_file", "study_id", "instrument_id"),
        default="source_file",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.attr_dim <= 0:
        parser.error("--attr-dim must be positive")

    output_paths = {
        split: args.output_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing split manifests: {existing}")

    records = load_manifest_jsonl(args.manifest, expected_attr_dim=args.attr_dim)
    splits = split_grouped_records(
        records,
        fractions={
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
        },
        group_by=args.group_by,
        seed=args.seed,
        expected_attr_dim=args.attr_dim,
    )
    for split_name, split_records in splits.items():
        save_manifest_jsonl(
            split_records,
            output_paths[split_name],
            expected_attr_dim=args.attr_dim,
        )

    print(
        json.dumps(
            {
                split: {
                    "samples": len(split_records),
                    "source_files": len({record.source_file for record in split_records}),
                    "path": str(output_paths[split]),
                }
                for split, split_records in splits.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
