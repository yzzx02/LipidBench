from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lipidbench.data import (
    BASE_ATTRIBUTE_NAMES,
    PeakManifestRecord,
    PeakMultiTaskDataset,
    assert_source_file_disjoint,
    audit_manifest_records,
    collate_peak_multitask_batch,
    load_manifest_jsonl,
    save_manifest_jsonl,
    split_grouped_records,
    validate_manifest_records,
)


def _record(
    index: int,
    boxes: list[list[float]],
    *,
    source_file: str,
    seed_label: int,
    missing_tpas: bool = False,
    subsets: list[str] | None = None,
) -> PeakManifestRecord:
    attributes: list[float | None] = [
        float(index + offset) for offset in range(len(BASE_ATTRIBUTE_NAMES))
    ]
    if missing_tpas:
        attributes[3] = None
    return PeakManifestRecord.from_mapping(
        {
            "sample_id": f"sample-{index}",
            "image_path": f"sample_{index}.png",
            "boxes": boxes,
            "seed_box": [8.0, 4.0, 20.0, 60.0],
            "seed_label": seed_label,
            "attributes": attributes,
            "source_file": source_file,
            "study_id": f"study-{index % 2}",
            "instrument_id": f"instrument-{index % 3}",
            "subsets": subsets or [],
            "metadata": {"synthetic": True, "batch": index % 2},
        }
    )


@pytest.fixture
def synthetic_records() -> list[PeakManifestRecord]:
    return [
        _record(0, [], source_file="source-a", seed_label=0, subsets=["low_snr"]),
        _record(
            1,
            [[38.0, 5.0, 52.0, 59.0]],
            source_file="source-a",
            seed_label=0,
            missing_tpas=True,
            subsets=["low_snr", "rt_shift"],
        ),
        _record(
            2,
            [[4.0, 5.0, 18.0, 58.0], [35.0, 6.0, 49.0, 57.0]],
            source_file="source-b",
            seed_label=1,
            subsets=["double_peak"],
        ),
        _record(
            3,
            [[20.0, 8.0, 36.0, 56.0]],
            source_file="source-c",
            seed_label=1,
            subsets=["shoulder"],
        ),
        _record(
            4,
            [],
            source_file="source-d",
            seed_label=0,
            missing_tpas=True,
            subsets=["tailing"],
        ),
        _record(
            5,
            [[6.0, 6.0, 17.0, 58.0], [40.0, 4.0, 55.0, 60.0]],
            source_file="source-e",
            seed_label=1,
            subsets=["double_peak", "tailing"],
        ),
    ]


def test_manifest_jsonl_round_trip_uses_only_synthetic_fixture(
    tmp_path: Path,
    synthetic_records: list[PeakManifestRecord],
) -> None:
    manifest_path = tmp_path / "synthetic_manifest.jsonl"
    save_manifest_jsonl(synthetic_records, manifest_path)
    loaded = load_manifest_jsonl(manifest_path)

    assert loaded == synthetic_records
    assert all(record.metadata["synthetic"] is True for record in loaded)


def test_dataset_is_lazy_and_returns_complete_multitask_sample(
    synthetic_records: list[PeakManifestRecord],
) -> None:
    opened: list[str] = []

    def synthetic_loader(path: Path) -> torch.Tensor:
        opened.append(path.name)
        return torch.full((3, 64, 64), 0.25, dtype=torch.float32)

    dataset = PeakMultiTaskDataset(
        synthetic_records,
        image_root="unused_synthetic_root",
        image_loader=synthetic_loader,
    )
    assert opened == []

    empty = dataset[0]
    independent_false_seed = dataset[1]
    multiple = dataset[2]

    assert opened == ["sample_0.png", "sample_1.png", "sample_2.png"]
    assert empty["target"]["boxes"].shape == (0, 4)
    assert empty["target"]["labels"].shape == (0,)
    assert independent_false_seed["seed_label"].item() == 0.0
    assert independent_false_seed["target"]["boxes"].shape == (1, 4)
    assert independent_false_seed["seed_box"].shape == (1, 4)
    assert independent_false_seed["attributes"].shape == (16,)
    assert independent_false_seed["attribute_mask"].shape == (16,)
    assert not independent_false_seed["attribute_mask"][3]
    assert torch.isnan(independent_false_seed["attributes"][3])
    assert multiple["target"]["boxes"].shape == (2, 4)
    assert torch.all(multiple["target"]["labels"] == 1)
    assert multiple["metadata"]["source_file"] == "source-b"
    assert multiple["metadata"]["synthetic"] is True

    batch = collate_peak_multitask_batch([empty, independent_false_seed, multiple])
    assert len(batch["images"]) == 3
    assert [target["boxes"].shape[0] for target in batch["targets"]] == [0, 1, 2]
    assert batch["attributes"].shape == (3, 16)
    assert batch["attribute_masks"].shape == (3, 16)
    assert batch["seed_labels"].shape == (3,)
    assert len(batch["seed_boxes"]) == 3


def test_dataset_remaps_foreign_windows_image_root(
    synthetic_records: list[PeakManifestRecord],
    tmp_path: Path,
) -> None:
    opened: list[Path] = []
    mapping = synthetic_records[0].to_mapping()
    mapping["image_path"] = (
        r"Z:\foreign\PeakTruthLab\datasets\eic_images_flat"
        r"\source-a\sample_0.png"
    )
    record = PeakManifestRecord.from_mapping(mapping)
    image_root = tmp_path / "eic_images_flat"

    def synthetic_loader(path: Path) -> torch.Tensor:
        opened.append(path)
        return torch.full((3, 64, 64), 0.25, dtype=torch.float32)

    dataset = PeakMultiTaskDataset(
        [record],
        image_root=image_root,
        image_loader=synthetic_loader,
    )
    dataset[0]

    assert opened == [image_root / "source-a" / "sample_0.png"]


def test_dataset_avoids_duplicate_root_for_portable_manifest(
    synthetic_records: list[PeakManifestRecord],
    tmp_path: Path,
) -> None:
    opened: list[Path] = []
    mapping = synthetic_records[0].to_mapping()
    mapping["image_path"] = "eic_images_flat/source-a/sample_0.png"
    record = PeakManifestRecord.from_mapping(mapping)
    image_root = tmp_path / "eic_images_flat"

    def synthetic_loader(path: Path) -> torch.Tensor:
        opened.append(path)
        return torch.full((3, 64, 64), 0.25, dtype=torch.float32)

    dataset = PeakMultiTaskDataset(
        [record],
        image_root=image_root,
        image_loader=synthetic_loader,
    )
    dataset[0]

    assert opened == [image_root / "source-a" / "sample_0.png"]


def test_grouped_split_is_reproducible_and_source_disjoint(
    synthetic_records: list[PeakManifestRecord],
) -> None:
    kwargs = {
        "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
        "group_by": "source_file",
        "seed": 17,
    }
    first = split_grouped_records(synthetic_records, **kwargs)
    second = split_grouped_records(synthetic_records, **kwargs)

    assert {
        split: [record.sample_id for record in records]
        for split, records in first.items()
    } == {
        split: [record.sample_id for record in records]
        for split, records in second.items()
    }
    assert sum(len(records) for records in first.values()) == len(synthetic_records)
    assert all(first[split] for split in ("train", "val", "test"))
    assert_source_file_disjoint(first)

    source_sets = [
        {record.source_file for record in first[split]}
        for split in ("train", "val", "test")
    ]
    assert source_sets[0].isdisjoint(source_sets[1])
    assert source_sets[0].isdisjoint(source_sets[2])
    assert source_sets[1].isdisjoint(source_sets[2])


def test_source_leakage_check_rejects_overlap(
    synthetic_records: list[PeakManifestRecord],
) -> None:
    with pytest.raises(ValueError, match="source_file leakage"):
        assert_source_file_disjoint(
            {
                "train": [synthetic_records[0]],
                "test": [synthetic_records[1]],
            }
        )


def test_audit_counts_geometry_missing_attributes_and_anchor_advice(
    synthetic_records: list[PeakManifestRecord],
) -> None:
    report = audit_manifest_records(synthetic_records)

    assert report["samples"] == {
        "total": 6,
        "unique_source_files": 5,
        "unique_studies": 2,
        "unique_instruments": 3,
    }
    assert report["detection_boxes"]["empty_images"] == 2
    assert report["detection_boxes"]["single_box_images"] == 2
    assert report["detection_boxes"]["multi_box_images"] == 2
    assert report["detection_boxes"]["total_boxes"] == 6
    assert report["seed_labels"] == {
        "positive": 3,
        "negative": 3,
        "positive_fraction": 0.5,
    }
    assert report["attributes"]["dimension"] == len(BASE_ATTRIBUTE_NAMES)
    assert report["attributes"]["missing_by_name"]["TPAS"] == 2
    assert report["attributes"]["total_missing"] == 2
    assert report["subsets"]["low_snr"] == 2
    assert report["subsets"]["double_peak"] == 2
    assert report["anchor_advice"]["advisory_only"] is True
    assert report["anchor_advice"]["suggested_anchor_sizes"]
    assert report["anchor_advice"]["suggested_aspect_ratios"]


def test_manifest_validation_rejects_duplicate_ids_and_wrong_attribute_count(
    synthetic_records: list[PeakManifestRecord],
) -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_manifest_records([synthetic_records[0], synthetic_records[0]])

    invalid = synthetic_records[0].to_mapping()
    invalid["attributes"] = invalid["attributes"][:-1]
    with pytest.raises(ValueError, match="16 values"):
        PeakManifestRecord.from_mapping(invalid)
