from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _find_pairs(source_root: Path):
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted([p for p in source_root.rglob("*") if p.suffix.lower() in exts])

    pairs: list[tuple[Path, Path]] = []
    for img in image_paths:
        js = img.with_suffix(".json")
        if js.exists():
            pairs.append((img, js))
    return pairs


def _copy_or_move_pair(img: Path, js: Path, src_root: Path, dst_subset_root: Path, move: bool):
    rel = img.parent.relative_to(src_root)
    dst_dir = dst_subset_root / rel
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_img = dst_dir / img.name
    dst_js = dst_dir / js.name

    if move:
        shutil.move(str(img), str(dst_img))
        shutil.move(str(js), str(dst_js))
    else:
        shutil.copy2(str(img), str(dst_img))
        shutil.copy2(str(js), str(dst_js))


def main():
    parser = argparse.ArgumentParser("Train/Val split for JPEG+LabelMe JSON pairs")
    parser.add_argument(
        "--source-root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\datasets\eic_images",
        help="Source root containing images and same-name JSON files",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\datasets\split",
        help="Output root. Will create train/ and val/",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    args = parser.parse_args()

    src_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    train_root = out_root / "train"
    val_root = out_root / "val"

    if not src_root.exists():
        raise FileNotFoundError(f"source root not found: {src_root}")

    pairs = _find_pairs(src_root)
    if not pairs:
        raise RuntimeError(f"No image+json pairs found under: {src_root}")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(round(n_total * float(args.val_ratio)))
    n_val = max(1, min(n_total - 1, n_val)) if n_total > 1 else 0

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for img, js in train_pairs:
        _copy_or_move_pair(img, js, src_root, train_root, move=args.move)

    for img, js in val_pairs:
        _copy_or_move_pair(img, js, src_root, val_root, move=args.move)

    print(f"source_root: {src_root}")
    print(f"out_root:    {out_root}")
    print(f"total:       {n_total}")
    print(f"train:       {len(train_pairs)}")
    print(f"val:         {len(val_pairs)}")
    print(f"mode:        {'move' if args.move else 'copy'}")


if __name__ == "__main__":
    main()
