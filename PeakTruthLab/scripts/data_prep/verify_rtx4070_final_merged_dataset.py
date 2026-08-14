from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd


ATTRS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG", "SYM", "MOD", "EDGE"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_shapes(payload: dict) -> str:
    return json.dumps(payload.get("shapes", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root).resolve()
    seed = pd.read_csv(root / "tables/seed_master_16attrs.csv", dtype=str, keep_default_na=False)
    peaks = pd.read_csv(root / "tables/peak_instances_master_16attrs.csv", dtype=str, keep_default_na=False)
    split_frames = {name: pd.read_csv(root / "splits" / f"{name}.csv", dtype=str, keep_default_na=False) for name in ("train", "val", "test")}

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(len(seed) == 19817, f"seed rows {len(seed)} != 19817")
    require(len(peaks) == 18915, f"peak rows {len(peaks)} != 18915")
    require(seed["seed_id"].nunique() == len(seed), "seed_id not unique")
    require(seed["image_id"].nunique() == len(seed), "image_id not unique")
    require(peaks["peak_instance_id"].nunique() == len(peaks), "peak_instance_id not unique")
    require(set(peaks["seed_id"]) <= set(seed["seed_id"]), "peak table references unknown seed_id")
    require(Counter(peaks["label"]) == {"True_Peak": 18815, "OUT_FIG": 100}, f"peak label counts wrong: {Counter(peaks['label'])}")
    require(Counter(seed["seed_label"]) == {"1": 11109, "0": 8708}, f"seed label counts wrong: {Counter(seed['seed_label'])}")

    path_issues = []
    missing_files = []
    hash_mismatches = []
    shape_counts = Counter()
    peak_shape_rows = []
    new_input_json_by_stem = {path.stem: path for path in Path(args.new_root).joinpath("corrected_annotations").rglob("*.json")}
    new_shape_mismatches = []
    for row in seed.itertuples(index=False):
        image_rel = str(row.image)
        json_rel = str(row.annotation_json)
        for value in (image_rel, json_rel):
            pure = PurePosixPath(value)
            if pure.is_absolute() or ":" in value or "\\" in value or ".." in pure.parts:
                path_issues.append(value)
        image_path = root / image_rel
        json_path = root / json_rel
        if not image_path.is_file() or not json_path.is_file():
            missing_files.append(str(row.seed_id))
            continue
        image_sha = sha256_file(image_path)
        json_sha = sha256_file(json_path)
        if image_sha != row.image_sha256 or json_sha != row.annotation_sha256:
            hash_mismatches.append(str(row.seed_id))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        require(payload.get("imagePath") == image_path.name, f"imagePath mismatch: {row.seed_id}")
        require(payload.get("imageData") is None, f"imageData not removed: {row.seed_id}")
        for shape in payload.get("shapes", []):
            shape_counts[str(shape.get("label"))] += 1
            points = shape.get("points", [])
            if len(points) >= 2:
                peak_shape_rows.append((str(row.seed_id), str(shape.get("label")), points))
        if str(row.seed_id).startswith("NEW::"):
            stem = str(row.original_seed_id)
            source_payload = json.loads(new_input_json_by_stem[stem].read_text(encoding="utf-8"))
            if canonical_shapes(source_payload) != canonical_shapes(payload):
                new_shape_mismatches.append(stem)

    require(not path_issues, f"non-portable paths: {path_issues[:5]}")
    require(not missing_files, f"missing image/json pairs: {missing_files[:5]}")
    require(not hash_mismatches, f"file hash mismatches: {hash_mismatches[:5]}")
    require(not new_shape_mismatches, f"new LabelMe shapes changed: {new_shape_mismatches[:5]}")
    require(shape_counts == {"True_Peak": 18815, "OUT_FIG": 100}, f"LabelMe shape counts wrong: {shape_counts}")

    png_count = sum(1 for _ in (root / "images").rglob("*.png"))
    json_count = sum(1 for _ in (root / "images").rglob("*.json"))
    require(png_count == 19817, f"PNG count {png_count} != 19817")
    require(json_count == 19817, f"LabelMe JSON count {json_count} != 19817")

    attr_report = {}
    for entity, frame in (("seed", seed), ("peak_instance", peaks)):
        attr_report[entity] = {}
        for attr in ATTRS:
            values = pd.to_numeric(frame[attr], errors="coerce")
            attr_report[entity][attr] = {
                "nan_count": int(values.isna().sum()),
                "inf_count": int(sum(math.isinf(value) for value in values.dropna())),
            }
            require(attr_report[entity][attr]["inf_count"] == 0, f"{entity} {attr} has inf")

    split_counts = {name: len(frame) for name, frame in split_frames.items()}
    require(split_counts == {"train": 15853, "val": 1982, "test": 1982}, f"split counts wrong: {split_counts}")
    for field in ("seed_id", "image_id", "image", "image_sha256", "split_group_id"):
        sets = {name: set(frame[field]) for name, frame in split_frames.items()}
        require(not (sets["train"] & sets["val"]), f"{field} train/val leakage")
        require(not (sets["train"] & sets["test"]), f"{field} train/test leakage")
        require(not (sets["val"] & sets["test"]), f"{field} val/test leakage")
    require(not split_frames["test"]["exclude_from_test"].str.lower().isin({"true", "1"}).any(), "excluded conflict row present in Test")

    lock = json.loads((root / "splits/test_manifest_lock.json").read_text(encoding="utf-8"))
    test_csv_sha = sha256_file(root / "splits/test.csv")
    test_jsonl_sha = sha256_file(root / "splits/test.jsonl")
    require(lock["test_csv_sha256"] == test_csv_sha, "locked test CSV SHA mismatch")
    require(lock["test_jsonl_sha256"] == test_jsonl_sha, "locked test JSONL SHA mismatch")

    protected_root = Path(args.old_root)
    protected_checks = {
        "train_manifest": (protected_root / "manifests/train.jsonl", "8163d3d4f57c322aaa6cf2dc7f45c495a52b84d654bf4354d6adbe29309036a7"),
        "val_manifest": (protected_root / "manifests/val.jsonl", "e3b461cb74d4a2743d74c94771133f3d7b29e8b50d54e016be18c02bc933c76c"),
        "test_manifest": (protected_root / "manifests/test.jsonl", "16c2c80106c688ef1e35e493fea06129ebd168cbfd28ca6ea293ef577d4b5a75"),
        "representative_first": (protected_root / "eic_images_flat/0001_MAY_RoCI-StEM_CP-287/0001_MAY_RoCI-StEM_CP-287__F10001.png", "ad33a3b978d8cce35bb762fbb4ad4b0c1082360f05c82b0e7a7fc447e36ee56b"),
        "representative_middle": (protected_root / "eic_images_flat/D03P_POS/D03P_POS__F1233.png", "8927b86b2d334aa4f3df2ff07c0cb6d097c3dfc85329da08f8593fb40ca08a7a"),
        "representative_last": (protected_root / "eic_images_flat/WTHFD_mixpos/WTHFD_mixpos__F9990.png", "d3cfc07acf6132de4d84f24cc3c591d9a95fa9e7385ee49ff9b6a33c2c1c1743"),
    }
    protected_actual = {}
    for name, (path, expected) in protected_checks.items():
        actual = sha256_file(path)
        protected_actual[name] = actual
        require(actual == expected, f"protected old-final hash changed: {name}")

    release_sha = sha256_file(Path(args.release_zip))
    require(release_sha.upper() == "F89632310064E5F92A484AB7FD75A05ABF3DC55631F7077B2C9A328D4F0CB409", "new release ZIP SHA mismatch")

    result = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "verified_counts": {
            "seed_rows": len(seed),
            "peak_rows": len(peaks),
            "png_files": png_count,
            "labelme_json_files": json_count,
            "labelme_shapes": dict(shape_counts),
            "split_rows": split_counts,
        },
        "attribute_nonfinite": attr_report,
        "portable_path_issues": len(path_issues),
        "missing_files": len(missing_files),
        "content_hash_mismatches": len(hash_mismatches),
        "new_labelme_shape_mismatches": len(new_shape_mismatches),
        "test_lock": {"test_csv_sha256": test_csv_sha, "test_jsonl_sha256": test_jsonl_sha},
        "protected_old_final_hashes": protected_actual,
        "new_release_zip_sha256": release_sha,
    }
    output = root / "audits/independent_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814")
    parser.add_argument("--old-root", default=r"D:\CODE\LipidBench\PeakTruthLab\results\paper_final_reviewed_20260725\dataset_release\PeakTruthLab-dataset-v2")
    parser.add_argument("--new-root", default=r"D:\CODE\downloads\PeakTruthLab_RTX4070_20260814\extracted\manual_negative_4500_v2_staging")
    parser.add_argument("--release-zip", default=r"D:\CODE\downloads\PeakTruthLab_RTX4070_20260814\PeakTruthLab_manual_negative_4500_v2_20260814.zip")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
