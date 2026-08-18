"""Build the public RTX4070 final release without cross-domain checkpoints.

Creates three auditable ZIP assets: Main model weights, all non-checkpoint
second-experiment results, and the complete final dataset. LODO/fold
checkpoints, last checkpoints, and the invalid batch-8/15-epoch run are
intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


RELEASE_NAME = "rtx4070_final_20260818"
FORMAL_RUN = "multitask_concat_detection_seed_20260816_bs16_ep30"
INVALID_RUN = "multitask_concat_detection_seed_20260815"
RESULT_DIRECTORIES = (
    "smoke",
    "main_ablation",
    "main_test_single_use_20260815",
    "sanity_audit_20260815",
    "lodo_concat_seed_20260814",
    FORMAL_RUN,
)
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


@dataclass(frozen=True)
class AssetRecord:
    file: str
    bytes: int
    sha256: str
    member_count: int
    purpose: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def write_zip(
    destination: Path,
    members: Iterable[tuple[Path, str]],
    *,
    store_checkpoints: bool = False,
) -> int:
    count = 0
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for source, archive_name in members:
            compression = (
                zipfile.ZIP_STORED
                if store_checkpoints and source.suffix.lower() in CHECKPOINT_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            archive.write(source, archive_name.replace(os.sep, "/"), compress_type=compression)
            count += 1
    return count


def weight_members(formal_root: Path) -> list[tuple[Path, str]]:
    training = formal_root / "main_split" / "training"
    evaluation = formal_root / "main_split" / "locked_test_evaluation"
    members = [
        (training / name, f"main_model/{name}")
        for name in (
            "best_detection.pt",
            "best_seed.pt",
            "attribute_preprocessing.json",
            "run_config.json",
            "summary.json",
            "history.json",
        )
    ]
    members.extend(
        (evaluation / name, f"main_model/locked_test_evaluation/{name}")
        for name in (
            "selection_before_target_access.json",
            "target_metrics.json",
            "evaluation_complete.json",
        )
    )
    missing = [str(path) for path, _ in members if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Main model files:\n" + "\n".join(missing))
    return members


def result_members(results_root: Path) -> Iterator[tuple[Path, str]]:
    for directory_name in RESULT_DIRECTORIES:
        source_root = results_root / directory_name
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing required result directory: {source_root}")
        for source in iter_files(source_root):
            if source.suffix.lower() in CHECKPOINT_SUFFIXES:
                continue
            yield source, str(Path("second_experiment_results") / source.relative_to(results_root))


def dataset_members(dataset_root: Path) -> Iterator[tuple[Path, str]]:
    for source in iter_files(dataset_root):
        yield source, str(Path(dataset_root.name) / source.relative_to(dataset_root))


def validate(formal_root: Path, results_root: Path, dataset_root: Path) -> None:
    state_path = formal_root / "PIPELINE_STATE.json"
    aggregate_path = formal_root / "multitask_concat_main_and_lodo_results.csv"
    if not state_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError("Formal run is missing final state or aggregate results")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError(f"Formal pipeline is not complete: {state.get('status')!r}")
    if INVALID_RUN in RESULT_DIRECTORIES:
        raise RuntimeError("Invalid batch-8/15 run must never be selected")
    verification_path = dataset_root / "audits" / "independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "ok" or verification.get("failures"):
        raise RuntimeError("Final dataset verification is not clean")
    counts = verification.get("verified_counts", {})
    expected = {"seed_rows": 19817, "peak_rows": 18915, "png_files": 19817}
    mismatches = {
        key: (counts.get(key), value)
        for key, value in expected.items()
        if counts.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Final dataset count mismatch: {mismatches}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "PeakTruthLab" / "releases" / RELEASE_NAME,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    results_root = repo_root / "PeakTruthLab" / "results" / "rtx4070_final_merged_20260814"
    formal_root = results_root / FORMAL_RUN
    dataset_root = repo_root / "PeakTruthLab" / "datasets" / "PeakTruthLab_final_merged_20260814"
    validate(formal_root, results_root, dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = (
        (
            output_dir / "LipidBench_Main_model_weights_20260818.zip",
            weight_members(formal_root),
            True,
            "Main split best_detection.pt and best_seed.pt with locked configuration",
        ),
        (
            output_dir / "LipidBench_second_experiment_results_no_fold_weights_20260818.zip",
            result_members(results_root),
            False,
            "All second-experiment metrics, histories, thresholds, predictions, logs, and audits; no checkpoints",
        ),
        (
            output_dir / "PeakTruthLab_final_dataset_20260814.zip",
            dataset_members(dataset_root),
            False,
            "Final merged portable dataset with PNG, LabelMe JSON, raw 16 attributes, splits, manifests, and QC",
        ),
    )

    records: list[AssetRecord] = []
    for destination, members, store_checkpoints, purpose in assets:
        print(f"Building {destination.name} ...", flush=True)
        member_count = write_zip(destination, members, store_checkpoints=store_checkpoints)
        records.append(
            AssetRecord(
                file=destination.name,
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
                member_count=member_count,
                purpose=purpose,
            )
        )

    manifest_json = output_dir / "release_manifest.json"
    manifest_csv = output_dir / "release_manifest.csv"
    manifest_json.write_text(
        json.dumps(
            {
                "release": RELEASE_NAME,
                "checkpoint_policy": {
                    "included": [
                        "main_split/training/best_detection.pt",
                        "main_split/training/best_seed.pt",
                    ],
                    "excluded": [
                        "all last.pt/last.pth",
                        "all LODO/fold checkpoints",
                        INVALID_RUN,
                    ],
                },
                "assets": [asdict(record) for record in records],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    print(json.dumps([asdict(record) for record in records], indent=2), flush=True)


if __name__ == "__main__":
    main()
