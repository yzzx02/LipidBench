from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.data import BASE_ATTRIBUTE_NAMES, audit_manifest_records, load_manifest_jsonl


def _attribute_names(attr_dim: int) -> tuple[str, ...]:
    if attr_dim == len(BASE_ATTRIBUTE_NAMES):
        return BASE_ATTRIBUTE_NAMES
    return tuple(f"attribute_{index + 1:02d}" for index in range(attr_dim))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a PeakMultiTaskRCNN JSONL manifest without changing model YAML.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--attr-dim", type=int, default=len(BASE_ATTRIBUTE_NAMES))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    if args.attr_dim <= 0:
        parser.error("--attr-dim must be positive")

    records = load_manifest_jsonl(args.manifest, expected_attr_dim=args.attr_dim)
    report = audit_manifest_records(records, attribute_names=_attribute_names(args.attr_dim))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
