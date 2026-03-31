from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ATTR_COLUMNS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG"]


def _image_relpath(source_file: str, feature_id: str) -> str:
    stem = str(source_file).replace(".mzML", "")
    return str(Path(stem) / f"{feature_id}.png")


def _parse_csv_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_float_range(raw: str) -> tuple[float, float]:
    txt = str(raw).strip()
    if "-" in txt:
        a, b = txt.split("-", 1)
    elif "," in txt:
        a, b = txt.split(",", 1)
    else:
        v = float(txt)
        return v, v
    lo, hi = float(a), float(b)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _choose_count(total_groups: int, ratio: float, min_n: int, max_n: int | None) -> int:
    n = int(round(total_groups * float(ratio)))
    n = max(min_n, n)
    if max_n is not None:
        n = min(max_n, n)
    return min(total_groups, n)


def _pick_groups(
    rng: np.random.Generator,
    candidates: list[str],
    target_n: int,
    forced: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    forced = forced or []
    forced_set = set(forced)
    if not forced_set.issubset(set(candidates)):
        missing = sorted(forced_set - set(candidates))
        raise ValueError(f"Forced groups not in candidate list: {missing}")

    selected = list(sorted(forced_set))
    need = max(0, int(target_n) - len(selected))
    pool = [g for g in candidates if g not in forced_set]
    if need > len(pool):
        need = len(pool)
    if need > 0:
        sampled = rng.choice(np.array(pool, dtype=object), size=need, replace=False).tolist()
        selected.extend(sampled)

    selected = sorted(selected)
    selected_set = set(selected)
    rest = [g for g in candidates if g not in selected_set]
    return selected, rest


def _assert_no_overlap(train_groups: set[str], val_groups: set[str], test_groups: set[str]) -> None:
    tv = sorted(train_groups & val_groups)
    tt = sorted(train_groups & test_groups)
    vt = sorted(val_groups & test_groups)
    if tv or tt or vt:
        raise RuntimeError(
            "Group overlap detected: "
            f"train∩val={tv}, train∩test={tt}, val∩test={vt}"
        )


def prepare(args: argparse.Namespace) -> None:
    csv_path = Path(args.input_csv).resolve()
    image_root = Path(args.image_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    df = pd.read_csv(csv_path)
    req = {"Feature_ID", "source_file", "is_true_peak", *ATTR_COLUMNS}
    miss = sorted(req - set(df.columns))
    if miss:
        raise ValueError(f"Missing columns in {csv_path}: {miss}")

    work = df[df["is_true_peak"].notna()].copy()
    work["is_true_peak"] = work["is_true_peak"].astype(int)
    work["image"] = [
        _image_relpath(src, fid)
        for src, fid in zip(work["source_file"].astype(str), work["Feature_ID"].astype(str), strict=False)
    ]
    work["image_abs"] = work["image"].map(lambda x: str((image_root / x).resolve()))
    work = work[work["image"].map(lambda x: (image_root / x).exists())].copy()

    if args.drop_all_nan_attrs:
        work = work[~work[ATTR_COLUMNS].isna().all(axis=1)].copy()

    all_groups = sorted(work["source_file"].astype(str).unique().tolist())
    total_groups = len(all_groups)
    if total_groups < 3:
        raise ValueError(f"Need at least 3 source_file groups, got {total_groups}")

    test_lo, test_hi = _parse_float_range(args.test_group_ratio_range)
    if test_lo <= 0 or test_hi >= 1:
        raise ValueError("--test-group-ratio-range must be within (0,1)")
    if args.val_group_ratio <= 0 or args.val_group_ratio >= 1:
        raise ValueError("--val-group-ratio must be within (0,1)")

    forced_exact = _parse_csv_list(args.force_test_source_files)
    forced_keyword = str(args.force_test_keyword).strip()
    forced_pattern = [g for g in all_groups if forced_keyword and forced_keyword.lower() in g.lower()]
    forced_test = sorted(set(forced_exact) | set(forced_pattern))

    missing_forced = sorted(set(forced_exact) - set(all_groups))
    if missing_forced:
        raise ValueError(f"--force-test-source-files contains unknown source_file: {missing_forced}")

    test_target_by_ratio = _choose_count(
        total_groups=total_groups,
        ratio=(test_lo + test_hi) / 2.0,
        min_n=max(1, len(forced_test)),
        max_n=args.test_group_max,
    )
    test_target = max(test_target_by_ratio, args.test_group_min, len(forced_test))
    test_target = min(test_target, total_groups - 2)

    rng = np.random.default_rng(int(args.seed))
    test_groups, remain_after_test = _pick_groups(
        rng=rng,
        candidates=all_groups,
        target_n=test_target,
        forced=forced_test,
    )

    val_target = _choose_count(
        total_groups=total_groups,
        ratio=float(args.val_group_ratio),
        min_n=1,
        max_n=None,
    )
    val_target = min(val_target, max(1, len(remain_after_test) - 1))
    val_groups, train_groups = _pick_groups(
        rng=rng,
        candidates=remain_after_test,
        target_n=val_target,
        forced=None,
    )

    train_set, val_set, test_set = set(train_groups), set(val_groups), set(test_groups)
    _assert_no_overlap(train_set, val_set, test_set)

    train_df = work[work["source_file"].astype(str).isin(train_set)].copy()
    val_df = work[work["source_file"].astype(str).isin(val_set)].copy()
    test_df = work[work["source_file"].astype(str).isin(test_set)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Empty split detected: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    group_manifest_path = out_dir / "group_split_manifest.csv"
    summary_path = out_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest_rows = []
    for g in train_groups:
        manifest_rows.append({"source_file": g, "split": "train"})
    for g in val_groups:
        manifest_rows.append({"source_file": g, "split": "val"})
    for g in test_groups:
        manifest_rows.append({"source_file": g, "split": "test"})
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["split", "source_file"]).reset_index(drop=True)
    manifest_df.to_csv(group_manifest_path, index=False)

    summary = {
        "input_csv": str(csv_path),
        "image_root": str(image_root),
        "rows_total_labeled_with_image": int(len(work)),
        "groups_total": int(total_groups),
        "seed": int(args.seed),
        "forced_test_keyword": forced_keyword,
        "forced_test_source_files": forced_exact,
        "forced_test_resolved": sorted(forced_test),
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_groups": {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        },
        "label_counts": {
            "train": {str(int(k)): int(v) for k, v in train_df["is_true_peak"].value_counts(dropna=False).to_dict().items()},
            "val": {str(int(k)): int(v) for k, v in val_df["is_true_peak"].value_counts(dropna=False).to_dict().items()},
            "test": {str(int(k)): int(v) for k, v in test_df["is_true_peak"].value_counts(dropna=False).to_dict().items()},
        },
        "integrity": {
            "train_intersect_val": sorted(train_set & val_set),
            "train_intersect_test": sorted(train_set & test_set),
            "val_intersect_test": sorted(val_set & test_set),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"input_csv:     {csv_path}")
    print(f"image_root:    {image_root}")
    print(f"out_dir:       {out_dir}")
    print(f"rows_total:    {len(work)}")
    print(f"groups_total:  {total_groups}")
    print(f"test_groups:   {len(test_groups)} -> {test_groups}")
    print(f"val_groups:    {len(val_groups)} -> {val_groups}")
    print(f"train_groups:  {len(train_groups)} -> {train_groups}")
    print(f"train_rows:    {len(train_df)}")
    print(f"val_rows:      {len(val_df)}")
    print(f"test_rows:     {len(test_df)}")
    print(f"train_labels:  {train_df['is_true_peak'].value_counts(dropna=False).to_dict()}")
    print(f"val_labels:    {val_df['is_true_peak'].value_counts(dropna=False).to_dict()}")
    print(f"test_labels:   {test_df['is_true_peak'].value_counts(dropna=False).to_dict()}")
    print(f"train_csv:     {train_path}")
    print(f"val_csv:       {val_path}")
    print(f"test_csv:      {test_path}")
    print(f"manifest_csv:  {group_manifest_path}")
    print(f"summary_json:  {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare grouped train/val/test CSVs for ConvNeXt fusion")
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--image-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--out-dir", type=str, default="PeakTruthLab/results/convnext_binary_split")
    p.add_argument(
        "--test-group-ratio-range",
        type=str,
        default="0.15-0.20",
        help="Target ratio range for test source_file groups, e.g. 0.15-0.20",
    )
    p.add_argument("--test-group-min", type=int, default=4, help="Minimum number of test source_file groups")
    p.add_argument("--test-group-max", type=int, default=5, help="Maximum number of test source_file groups")
    p.add_argument("--val-group-ratio", type=float, default=0.15, help="Validation source_file group ratio")
    p.add_argument(
        "--force-test-keyword",
        type=str,
        default="Sphingolipid",
        help="Any source_file containing this keyword will be forced into test split",
    )
    p.add_argument(
        "--force-test-source-files",
        type=str,
        default="",
        help="Optional comma-separated exact source_file names forced into test split",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-all-nan-attrs", action="store_true")
    args = p.parse_args()
    if args.test_group_min < 1:
        raise ValueError("--test-group-min must be >= 1")
    if args.test_group_max < args.test_group_min:
        raise ValueError("--test-group-max must be >= --test-group-min")
    return args


if __name__ == "__main__":
    prepare(parse_args())
