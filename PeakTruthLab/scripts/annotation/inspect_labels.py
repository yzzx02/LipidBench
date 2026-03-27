from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_label_map(label_map_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    with label_map_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    labels: dict[str, int] = {str(k): int(v) for k, v in (obj.get("labels") or {}).items()}
    aliases: dict[str, str] = {str(k): str(v) for k, v in (obj.get("aliases") or {}).items()}
    if not labels:
        raise ValueError(f"label_map has empty labels: {label_map_path}")
    if "True_Peak" not in labels:
        raise ValueError("label_map.json must include 'True_Peak'")
    return labels, aliases


def main() -> int:
    p = argparse.ArgumentParser("Inspect LabelMe labels distribution")
    p.add_argument(
        "--data-root",
        type=str,
        default=r"D:\\LipidBench\\PeakTruthLab\\datasets\\small_trainset",
        help="Folder with images and same-name LabelMe JSON",
    )
    p.add_argument(
        "--label-map",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.json"),
    )
    args = p.parse_args()

    root = Path(args.data_root)
    label_map_path = Path(args.label_map)
    labels, aliases = _load_label_map(label_map_path)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise RuntimeError(f"no images found under: {root}")

    counts = Counter()
    empty = 0
    missing_json = 0
    unknown = Counter()

    for img in images:
        js = img.with_suffix(".json")
        if not js.exists():
            missing_json += 1
            continue
        with js.open("r", encoding="utf-8") as f:
            ann = json.load(f)
        shapes = ann.get("shapes") or []
        if len(shapes) == 0:
            empty += 1
            continue
        for s in shapes:
            name = str(s.get("label", "")).strip()
            if name in aliases:
                name = aliases[name]
            if name not in labels:
                unknown[name] += 1
            else:
                counts[name] += 1

    print("data_root:", root)
    print("images:", len(images))
    print("missing_json:", missing_json)
    print("empty_shapes (negative samples):", empty)
    print("\nlabel_counts:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    if unknown:
        print("\nunknown_labels (need add to label_map.json aliases/labels):")
        for k, v in unknown.most_common():
            print(f"  {k!r}: {v}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
