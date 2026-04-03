from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _load_label_map(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    labels = {str(k): int(v) for k, v in (obj.get("labels") or {}).items()}
    aliases = {str(k): str(v) for k, v in (obj.get("aliases") or {}).items()}
    return labels, aliases


def _normalize_label(name: str, aliases: dict[str, str]) -> str:
    n = str(name).strip()
    return aliases.get(n, n)


def _iter_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _extract_labels_from_json(json_path: Path, aliases: dict[str, str]) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    shapes = obj.get("shapes") or []
    out: list[str] = []
    for s in shapes:
        if not isinstance(s, dict):
            continue
        lb = s.get("label")
        if isinstance(lb, str) and lb.strip():
            out.append(_normalize_label(lb, aliases))
    return out


def main() -> int:
    p = argparse.ArgumentParser("Convert LabelMe annotations to binary CSV for true/false peak training")
    p.add_argument("--data-root", type=str, required=True, help="Folder with images and same-name LabelMe JSON")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    p.add_argument(
        "--label-map",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.json"),
    )
    p.add_argument(
        "--positive-labels",
        type=str,
        default="True_Peak,True_Peak_Coelution,True_Peak_Jagged",
        help="Comma-separated labels treated as positive class",
    )
    p.add_argument(
        "--unknown-policy",
        choices=["skip", "false", "error"],
        default="skip",
        help="How to handle samples with only unknown labels",
    )
    p.add_argument(
        "--export-folders",
        type=str,
        default=None,
        help="Optional folder to copy images into true/false subfolders for manual review",
    )

    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    label_map = Path(args.label_map).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not label_map.exists():
        raise FileNotFoundError(f"label_map not found: {label_map}")

    known_labels, aliases = _load_label_map(label_map)
    positive_labels = {x.strip() for x in args.positive_labels.split(",") if x.strip()}

    images = _iter_images(data_root)
    if not images:
        raise RuntimeError(f"No images found under: {data_root}")

    rows: list[dict[str, object]] = []
    n_missing_json = 0
    n_unknown_only = 0

    export_root = Path(args.export_folders).resolve() if args.export_folders else None
    if export_root is not None:
        (export_root / "true").mkdir(parents=True, exist_ok=True)
        (export_root / "false").mkdir(parents=True, exist_ok=True)

    for img in images:
        js = img.with_suffix(".json")
        if not js.exists():
            n_missing_json += 1
            continue

        labels = _extract_labels_from_json(js, aliases)

        if len(labels) == 0:
            y = 0
        else:
            known = [lb for lb in labels if lb in known_labels]
            if not known:
                n_unknown_only += 1
                if args.unknown_policy == "skip":
                    continue
                if args.unknown_policy == "error":
                    raise ValueError(f"Unknown-only labels in: {js}")
                y = 0
            else:
                y = 1 if any(lb in positive_labels for lb in known) else 0

        rel = img.relative_to(data_root).as_posix()
        rows.append({"image": rel, "is_true_peak": int(y)})

        if export_root is not None:
            target = export_root / ("true" if y == 1 else "false") / img.name
            shutil.copy2(str(img), str(target))

    if not rows:
        raise RuntimeError("No rows generated. Check labels and unknown-policy")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"data_root: {data_root}")
    print(f"out_csv:   {out_csv}")
    print(f"samples:   {len(df)}")
    print("class_dist:", df["is_true_peak"].value_counts().to_dict())
    print(f"missing_json_skipped: {n_missing_json}")
    print(f"unknown_only:         {n_unknown_only}")
    if export_root is not None:
        print(f"export_folders: {export_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
