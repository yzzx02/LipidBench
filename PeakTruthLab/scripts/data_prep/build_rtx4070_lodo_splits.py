from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ATTRS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG", "SYM", "MOD", "EDGE"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def describe(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "positive": int(frame["seed_label"].sum()),
        "negative": int(len(frame) - frame["seed_label"].sum()),
        "positive_rate": float(frame["seed_label"].mean()),
        "domains": int(frame["domain_id"].nunique()),
        "source_files": int(frame["source_file"].nunique()),
        "old": int((frame["old_new_batch"] == "old_final_reviewed_20260725").sum()),
        "new": int((frame["old_new_batch"] == "new_manual_negative_4500_v2_20260814").sum()),
    }


def main(args: argparse.Namespace) -> None:
    master_path = Path(args.master_csv).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty LODO split root: {output}")
    output.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path)
    required = {
        "image_id",
        "seed_id",
        "image",
        "annotation_json",
        "seed_label",
        "domain_id",
        "source_file",
        "old_new_batch",
        "image_sha256",
        "split_group_id",
        *ATTRS,
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(master) != 19817 or master["seed_id"].duplicated().any() or master["image_id"].duplicated().any():
        raise RuntimeError("Master row count or unique ID check failed")
    master["seed_label"] = master["seed_label"].astype(int)
    master["split_group_id"] = master["split_group_id"].fillna(master["seed_id"]).astype(str)

    group_domains = master.groupby("split_group_id")["domain_id"].nunique()
    cross_domain_groups = group_domains[group_domains > 1]
    if len(cross_domain_groups):
        raise RuntimeError(f"Duplicate/split groups cross domain boundaries: {len(cross_domain_groups)}")

    domains = sorted(master["domain_id"].dropna().astype(str).unique())
    if len(domains) != 11:
        raise RuntimeError(f"Expected 11 independent domains, found {len(domains)}: {domains}")

    fold_rows = []
    all_heldout_ids: set[str] = set()
    for fold_index, heldout_domain in enumerate(domains, start=1):
        fold_id = f"fold_{fold_index:02d}_{heldout_domain}"
        fold_dir = output / fold_id
        fold_dir.mkdir(parents=True)
        heldout = master.loc[master["domain_id"].astype(str) == heldout_domain].copy()
        development = master.loc[master["domain_id"].astype(str) != heldout_domain].copy()

        stratify = (
            development["domain_id"].fillna("<NA>").astype(str)
            + "||"
            + development["seed_label"].astype(str)
            + "||"
            + development["old_new_batch"].fillna("<NA>").astype(str)
        )
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=int(args.seed))
        train_index, val_index = next(splitter.split(development, stratify, groups=development["split_group_id"]))
        train = development.iloc[train_index].copy()
        val = development.iloc[val_index].copy()

        sets = {
            "train": train,
            "val": val,
            "heldout_test": heldout,
        }
        intersection_counts: dict[str, dict[str, int]] = {}
        for column in ("image_id", "seed_id", "image", "annotation_json", "image_sha256", "split_group_id"):
            values = {name: set(frame[column].dropna().astype(str)) - {""} for name, frame in sets.items()}
            intersection_counts[column] = {
                "train_val": len(values["train"] & values["val"]),
                "train_heldout": len(values["train"] & values["heldout_test"]),
                "val_heldout": len(values["val"] & values["heldout_test"]),
            }
            if any(intersection_counts[column].values()):
                raise RuntimeError(f"{fold_id}: leakage intersection in {column}: {intersection_counts[column]}")

        if set(train["domain_id"].astype(str)) & {heldout_domain} or set(val["domain_id"].astype(str)) & {heldout_domain}:
            raise RuntimeError(f"{fold_id}: held-out domain leaked into Train/Val")

        for name, frame in sets.items():
            csv_path = fold_dir / f"{name}.csv"
            jsonl_path = fold_dir / f"{name}.jsonl"
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
            frame.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        audit = {
            "status": "ok",
            "fold_id": fold_id,
            "heldout_domain": heldout_domain,
            "seed": int(args.seed),
            "development_split": "StratifiedGroupKFold(n_splits=10, shuffle=True); first fold used as Val",
            "train": describe(train),
            "val": describe(val),
            "heldout_test": describe(heldout),
            "intersection_counts": intersection_counts,
            "heldout_domain_absent_from_train_val": True,
            "preprocessing_policy": "Fit median imputation and population z-score only on this fold's Train",
            "checkpoint_threshold_policy": "Checkpoint and threshold selected only on this fold's Val",
            "heldout_policy": "Held-out domain is evaluated only after checkpoint and threshold are locked",
            "sha256": {
                name: {
                    "csv": sha256(fold_dir / f"{name}.csv"),
                    "jsonl": sha256(fold_dir / f"{name}.jsonl"),
                }
                for name in sets
            },
        }
        write_json(fold_dir / "split_audit.json", audit)
        (fold_dir / "LOCKED_HELDOUT_TEST.sha256").write_text(
            f"{audit['sha256']['heldout_test']['csv']}  heldout_test.csv\n{audit['sha256']['heldout_test']['jsonl']}  heldout_test.jsonl\n",
            encoding="utf-8",
        )
        fold_rows.append(
            {
                "fold_index": fold_index,
                "fold_id": fold_id,
                "heldout_domain": heldout_domain,
                "train_rows": len(train),
                "train_positive": int(train["seed_label"].sum()),
                "train_negative": int(len(train) - train["seed_label"].sum()),
                "val_rows": len(val),
                "val_positive": int(val["seed_label"].sum()),
                "val_negative": int(len(val) - val["seed_label"].sum()),
                "heldout_rows": len(heldout),
                "heldout_positive": int(heldout["seed_label"].sum()),
                "heldout_negative": int(len(heldout) - heldout["seed_label"].sum()),
                "heldout_csv_sha256": audit["sha256"]["heldout_test"]["csv"],
            }
        )
        overlap = all_heldout_ids & set(heldout["seed_id"].astype(str))
        if overlap:
            raise RuntimeError(f"Held-out samples repeated across folds: {len(overlap)}")
        all_heldout_ids.update(heldout["seed_id"].astype(str))

    if all_heldout_ids != set(master["seed_id"].astype(str)):
        raise RuntimeError("Held-out folds do not cover every master Seed exactly once")
    pd.DataFrame(fold_rows).to_csv(output / "LODO_FOLD_MANIFEST.csv", index=False, encoding="utf-8-sig")
    overall = {
        "status": "ok",
        "master_csv": str(master_path),
        "master_csv_sha256": sha256(master_path),
        "seed": int(args.seed),
        "domains": domains,
        "folds": len(fold_rows),
        "master_rows": len(master),
        "heldout_union_rows": len(all_heldout_ids),
        "each_seed_held_out_exactly_once": True,
        "cross_domain_duplicate_groups": 0,
        "test_accessed_for_model_selection": False,
    }
    write_json(output / "LODO_SPLIT_GLOBAL_AUDIT.json", overall)
    print(json.dumps({"overall": overall, "folds": fold_rows}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-csv", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\tables\seed_master_16attrs.csv")
    parser.add_argument("--output", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\lodo_seed_20260814")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
