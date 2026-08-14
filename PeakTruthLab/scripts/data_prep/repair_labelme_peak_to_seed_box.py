from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore one LabelMe True_Peak rectangle to its original Seed box."
    )
    parser.add_argument("--seed-table", type=Path, required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    seed_table = args.seed_table.resolve()
    feature_id = str(args.feature_id)
    backup_root = args.backup_root.resolve()
    audit_json = args.audit_json.resolve()

    table = pd.read_csv(seed_table)
    rows = table.loc[table["原始特征编号"].astype(str).eq(feature_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one seed row for {feature_id}, got {len(rows)}")
    row = rows.iloc[0]

    json_path = Path(str(row["原始标注JSON路径"])).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    shapes = payload.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        raise ValueError(f"Expected exactly one shape in {json_path}, got {len(shapes or [])}")
    shape = shapes[0]
    if shape.get("label") != "True_Peak" or shape.get("shape_type") != "rectangle":
        raise ValueError(f"Expected one True_Peak rectangle in {json_path}")

    old_points = shape.get("points")
    new_points = [
        [float(row["原始Seed框_xmin_px"]), float(row["原始Seed框_ymin_px"])],
        [float(row["原始Seed框_xmax_px"]), float(row["原始Seed框_ymax_px"])],
    ]
    if not (new_points[0][0] < new_points[1][0] and new_points[0][1] < new_points[1][1]):
        raise ValueError(f"Invalid Seed rectangle: {new_points}")

    before_hash = sha256(json_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / f"{json_path.stem}__before_seed_restore_{timestamp}.json"
    if backup_path.exists():
        raise FileExistsError(backup_path)
    shutil.copy2(json_path, backup_path)

    shape["points"] = new_points
    temp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    json.loads(temp_path.read_text(encoding="utf-8"))
    temp_path.replace(json_path)
    after_hash = sha256(json_path)
    if before_hash == after_hash:
        raise RuntimeError("Repair did not change the annotation JSON")

    audit = {
        "feature_id": feature_id,
        "json_path": str(json_path),
        "backup_path": str(backup_path),
        "seed_table": str(seed_table),
        "old_points": old_points,
        "restored_seed_points": new_points,
        "original_seed_rt": float(row["原始种子RT"]),
        "original_seed_left_rt": float(row["原始左边界RT"]),
        "original_seed_right_rt": float(row["原始右边界RT"]),
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "backup_sha256": sha256(backup_path),
    }
    if audit["backup_sha256"] != before_hash:
        raise RuntimeError("Backup hash does not match the pre-repair annotation")
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
