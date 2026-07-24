from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any


SPLITS = ("train", "val", "test")
PACKAGE_NAME = "PeakTruthLab-dataset-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            records.append(value)
    return records


def _image_suffix(image_path: str, image_root_name: str) -> Path:
    windows_path = PureWindowsPath(image_path)
    matching_indices = [
        index
        for index, part in enumerate(windows_path.parts)
        if part.casefold() == image_root_name.casefold()
    ]
    if matching_indices:
        suffix = windows_path.parts[matching_indices[-1] + 1 :]
        return Path(*suffix)
    path = Path(image_path)
    if path.is_absolute():
        raise ValueError(
            f"absolute image path does not contain {image_root_name!r}: "
            f"{image_path!r}"
        )
    if path.parts and path.parts[0].casefold() == image_root_name.casefold():
        return Path(*path.parts[1:])
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_dir = args.manifest_dir.resolve()
    image_root = args.image_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")

    split_records = {
        split: _read_jsonl(manifest_dir / f"{split}.jsonl")
        for split in SPLITS
    }
    sample_ids: set[str] = set()
    image_suffixes: dict[str, Path] = {}
    for split, records in split_records.items():
        for record in records:
            sample_id = str(record["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id across splits: {sample_id}")
            sample_ids.add(sample_id)
            suffix = _image_suffix(
                str(record["image_path"]),
                image_root.name,
            )
            source = image_root / suffix
            if not source.is_file():
                raise FileNotFoundError(
                    f"{split} sample {sample_id!r} image is missing: {source}"
                )
            image_suffixes[sample_id] = suffix

    with tempfile.TemporaryDirectory(
        prefix="peaktruthlab_dataset_",
        dir=output.parent,
    ) as temporary_directory:
        package_root = Path(temporary_directory) / PACKAGE_NAME
        packaged_images = package_root / "eic_images_flat"
        packaged_manifests = package_root / "manifests"
        packaged_manifests.mkdir(parents=True)

        for split, records in split_records.items():
            destination = packaged_manifests / f"{split}.jsonl"
            with destination.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    portable = dict(record)
                    suffix = image_suffixes[str(record["sample_id"])]
                    portable["image_path"] = suffix.as_posix()
                    handle.write(
                        json.dumps(
                            portable,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )

        for index, (sample_id, suffix) in enumerate(
            sorted(image_suffixes.items()),
            start=1,
        ):
            source = image_root / suffix
            destination = packaged_images / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "copy_images",
                            "completed": index,
                            "total": len(image_suffixes),
                        }
                    ),
                    flush=True,
                )

        info = {
            "package": PACKAGE_NAME,
            "format_version": 1,
            "image_path_base": "eic_images_flat",
            "splits": {
                split: len(records)
                for split, records in split_records.items()
            },
            "unique_samples": len(sample_ids),
            "unique_images": len(image_suffixes),
            "test_set_included": True,
            "test_set_used_during_training": False,
        }
        (package_root / "dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            files = sorted(
                path
                for path in package_root.rglob("*")
                if path.is_file()
            )
            for index, path in enumerate(files, start=1):
                archive.write(
                    path,
                    arcname=path.relative_to(package_root.parent),
                )
                if index % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "compress",
                                "completed": index,
                                "total": len(files),
                            }
                        ),
                        flush=True,
                    )

    with zipfile.ZipFile(output, mode="r") as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"archive CRC verification failed: {corrupt_member}")

    digest = _sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{digest}  {output.name}\n",
        encoding="ascii",
    )
    result = {
        **info,
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "sha256": digest,
        "checksum_file": str(checksum_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable PeakTruthLab train/val/test ZIP archive."
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    package_dataset(parse_args())
