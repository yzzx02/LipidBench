from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REMAP: dict[str, str] = {
    # Old class names (route B previously used)
    "True_Peak_HQ": "True_Peak",
    "True_Peak_Tailing": "True_Peak",
    # Common short / CN labels seen in the wild
    "HQ": "True_Peak",
    "高质量峰": "True_Peak",
    "拖尾峰": "True_Peak",
    "Tailing": "True_Peak",
}


def _iter_json_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _rewrite_one(path: Path, remap: dict[str, str], *, in_place: bool, backup_ext: str | None) -> tuple[bool, int]:
    """Returns (changed, num_labels_changed)."""

    with path.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {path} ({e})") from e

    shapes = obj.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        return False, 0

    changed = False
    n_changed = 0
    for s in shapes:
        if not isinstance(s, dict):
            continue
        label = s.get("label")
        if not isinstance(label, str):
            continue
        label_str = label.strip()
        if label_str in remap:
            s["label"] = remap[label_str]
            changed = True
            n_changed += 1

    if changed and in_place:
        if backup_ext:
            bak = path.with_suffix(path.suffix + backup_ext)
            if not bak.exists():
                bak.write_bytes(path.read_bytes())

        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")

    return changed, n_changed


def main() -> int:
    parser = argparse.ArgumentParser(
        "Remap LabelMe JSON shape labels in-place (e.g., HQ/Tailing -> True_Peak)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\datasets\small_trainset",
        help="Folder containing LabelMe JSON files (recursively).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write changes back to files (default is dry-run).",
    )
    parser.add_argument(
        "--backup-ext",
        type=str,
        default=".bak",
        help="Backup extension, e.g. '.bak'. Set to '' to disable backups.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    backup_ext = args.backup_ext
    if backup_ext is not None and backup_ext.strip() == "":
        backup_ext = None

    json_files = _iter_json_files(root)
    if not json_files:
        raise RuntimeError(f"No .json files found under: {root}")

    total_files = 0
    changed_files = 0
    total_labels_changed = 0

    for p in json_files:
        total_files += 1
        changed, n = _rewrite_one(p, DEFAULT_REMAP, in_place=bool(args.in_place), backup_ext=backup_ext)
        if changed:
            changed_files += 1
            total_labels_changed += n

    mode = "in-place" if args.in_place else "dry-run"
    print(f"root:                 {root}")
    print(f"mode:                 {mode}")
    print(f"json_files:            {total_files}")
    print(f"changed_files:         {changed_files}")
    print(f"labels_remapped_total: {total_labels_changed}")

    if not args.in_place:
        print("NOTE: No files were modified (dry-run). Re-run with --in-place to apply changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
