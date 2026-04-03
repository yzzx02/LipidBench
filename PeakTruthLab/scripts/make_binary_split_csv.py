from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _from_folder_layout(image_root: Path, true_dir_name: str, false_dir_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    true_dir = image_root / true_dir_name
    false_dir = image_root / false_dir_name

    if not true_dir.exists() or not false_dir.exists():
        raise FileNotFoundError(
            f"Folder mode requires both folders: {true_dir} and {false_dir}"
        )

    for label_dir, label in [(true_dir, 1), (false_dir, 0)]:
        for p in sorted(label_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "image": p.relative_to(image_root).as_posix(),
                        "is_true_peak": label,
                    }
                )

    if not rows:
        raise RuntimeError("No images found in folder mode")

    return pd.DataFrame(rows)


def _from_label_csv(labels_csv: Path, image_root: Path, image_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    missing = [c for c in [image_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {labels_csv}: {missing}")

    # 统一列名（保留其余属性列用于融合模型）
    if image_col != "image":
        df = df.rename(columns={image_col: "image"})
    if label_col != "is_true_peak":
        df = df.rename(columns={label_col: "is_true_peak"})

    # 只保留存在的图片
    valid_mask = []
    for rel in df["image"].astype(str).tolist():
        p = (image_root / Path(rel)).resolve()
        valid_mask.append(p.exists())
    df = df.loc[valid_mask].copy()

    if df.empty:
        raise RuntimeError("No valid image rows left after existence check")

    # 标签标准化为 0/1
    df["is_true_peak"] = df["is_true_peak"].astype(int)

    return df


def _split(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")

    y = df["is_true_peak"].astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=y,
    )

    if len(temp_df) == 0:
        return train_df, temp_df, temp_df

    if test_ratio == 0:
        return train_df, temp_df, temp_df.iloc[0:0]

    temp_test_ratio = test_ratio / (val_ratio + test_ratio)
    temp_y = temp_df["is_true_peak"].astype(int)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=temp_test_ratio,
        random_state=seed,
        stratify=temp_y,
    )
    return train_df, val_df, test_df


def main() -> None:
    p = argparse.ArgumentParser("Create train/val/test CSV for binary peak classification")

    p.add_argument("--image-root", type=str, required=True, help="Root dir used by --image-root in training")
    p.add_argument("--out-dir", type=str, default=r"D:\LipidBench\PeakTruthLab\datasets", help="Where to save train/val/test CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)

    # 两种输入模式：folder / csv（二选一）
    p.add_argument("--mode", choices=["folder", "csv"], default="csv")

    # folder mode
    p.add_argument("--true-dir-name", type=str, default="true")
    p.add_argument("--false-dir-name", type=str, default="false")

    # csv mode
    p.add_argument("--labels-csv", type=str, default=None, help="CSV with at least image,label and optional attr columns")
    p.add_argument("--image-col", type=str, default="image")
    p.add_argument("--label-col", type=str, default="is_true_peak")

    args = p.parse_args()

    image_root = Path(args.image_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "folder":
        df = _from_folder_layout(image_root, args.true_dir_name, args.false_dir_name)
    else:
        if not args.labels_csv:
            raise ValueError("--labels-csv is required in csv mode")
        df = _from_label_csv(Path(args.labels_csv).resolve(), image_root, args.image_col, args.label_col)

    train_df, val_df, test_df = _split(df, args.val_ratio, args.test_ratio, args.seed)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved split CSV files:")
    print(f"  train: {train_path} ({len(train_df)})")
    print(f"  val:   {val_path} ({len(val_df)})")
    print(f"  test:  {test_path} ({len(test_df)})")
    print("Label distribution:")
    print("  train:", train_df["is_true_peak"].value_counts().to_dict())
    print("  val:  ", val_df["is_true_peak"].value_counts().to_dict())
    print("  test: ", test_df["is_true_peak"].value_counts().to_dict())


if __name__ == "__main__":
    main()
