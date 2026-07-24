from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .peak_manifest import BASE_ATTRIBUTE_NAMES, PeakManifestRecord, validate_manifest_records


def assert_source_file_disjoint(
    splits: Mapping[str, Sequence[PeakManifestRecord]],
) -> None:
    owners: dict[str, str] = {}
    leaks: dict[str, set[str]] = defaultdict(set)
    for split_name, records in splits.items():
        for record in records:
            previous = owners.setdefault(record.source_file, split_name)
            if previous != split_name:
                leaks[record.source_file].update((previous, split_name))
    if leaks:
        details = {source: sorted(names) for source, names in sorted(leaks.items())}
        raise ValueError(f"source_file leakage across splits: {details}")


def split_grouped_records(
    records: Sequence[PeakManifestRecord],
    *,
    fractions: Mapping[str, float] | None = None,
    group_by: str = "source_file",
    seed: int = 42,
    expected_attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
) -> dict[str, list[PeakManifestRecord]]:
    """Split whole groups and always reject cross-split ``source_file`` leakage."""

    materialised = validate_manifest_records(
        records,
        expected_attr_dim=expected_attr_dim,
    )
    split_fractions = dict(fractions or {"train": 0.7, "val": 0.15, "test": 0.15})
    if not split_fractions:
        raise ValueError("fractions must define at least one split")
    if any(not math.isfinite(value) or value <= 0 for value in split_fractions.values()):
        raise ValueError("all split fractions must be finite and positive")
    total_fraction = sum(split_fractions.values())
    split_fractions = {
        name: value / total_fraction for name, value in split_fractions.items()
    }
    if group_by not in {"source_file", "study_id", "instrument_id"}:
        raise ValueError("group_by must be source_file, study_id, or instrument_id")

    grouped: dict[str, list[PeakManifestRecord]] = defaultdict(list)
    for record in materialised:
        grouped[str(getattr(record, group_by))].append(record)
    if len(grouped) < len(split_fractions):
        raise ValueError(
            f"need at least {len(split_fractions)} distinct {group_by} groups, "
            f"got {len(grouped)}"
        )

    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = sorted(
        split_fractions,
        key=lambda name: (-split_fractions[name], name),
    )
    result = {name: [] for name in split_fractions}
    counts = {name: 0 for name in split_fractions}
    target_counts = {
        name: split_fractions[name] * len(materialised) for name in split_fractions
    }

    for index, (_, group_records) in enumerate(groups):
        if index < len(split_names):
            destination = split_names[index]
        else:
            destination = min(
                split_names,
                key=lambda name: (
                    counts[name] / target_counts[name],
                    counts[name],
                    name,
                ),
            )
        result[destination].extend(group_records)
        counts[destination] += len(group_records)

    for split_records in result.values():
        split_records.sort(key=lambda record: record.sample_id)
    assert_source_file_disjoint(result)
    return result


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)
    result: dict[str, float] = {}
    for name, probability in (
        ("p05", 0.05),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p95", 0.95),
    ):
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = ordered[lower]
        else:
            weight = position - lower
            value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        result[name] = round(value, 6)
    return result


def _unique(values: Sequence[int | float]) -> list[int | float]:
    result: list[int | float] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _anchor_advice(
    areas: Sequence[float],
    aspect_ratios: Sequence[float],
) -> dict[str, Any]:
    scale_quantiles = _quantiles([math.sqrt(area) for area in areas])
    ratio_quantiles = _quantiles(aspect_ratios)
    suggested_sizes = _unique(
        [max(4, int(round(scale_quantiles[name] / 4.0) * 4)) for name in scale_quantiles]
    )
    suggested_ratios = _unique(
        [round(ratio_quantiles[name], 2) for name in ratio_quantiles]
    )
    return {
        "advisory_only": True,
        "ratio_definition": "height/width (TorchVision convention)",
        "suggested_anchor_sizes": suggested_sizes,
        "suggested_aspect_ratios": suggested_ratios,
        "note": "Review on training data; this report never edits the formal YAML.",
    }


def audit_manifest_records(
    records: Sequence[PeakManifestRecord],
    *,
    attribute_names: Sequence[str] = BASE_ATTRIBUTE_NAMES,
) -> dict[str, Any]:
    materialised = validate_manifest_records(
        records,
        expected_attr_dim=len(attribute_names),
    )
    box_counts = [len(record.boxes) for record in materialised]
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    for record in materialised:
        for x1, y1, x2, y2 in record.boxes:
            width = x2 - x1
            height = y2 - y1
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            aspect_ratios.append(height / width)

    seed_positive = sum(record.seed_label == 1 for record in materialised)
    seed_negative = len(materialised) - seed_positive
    missing_counts = {
        name: sum(record.attributes[index] is None for record in materialised)
        for index, name in enumerate(attribute_names)
    }
    total_attribute_values = len(materialised) * len(attribute_names)
    total_missing = sum(missing_counts.values())

    subset_counts: dict[str, int] = defaultdict(int)
    for record in materialised:
        for subset in set(record.subsets):
            subset_counts[subset] += 1

    return {
        "samples": {
            "total": len(materialised),
            "unique_source_files": len({record.source_file for record in materialised}),
            "unique_studies": len({record.study_id for record in materialised}),
            "unique_instruments": len({record.instrument_id for record in materialised}),
        },
        "detection_boxes": {
            "empty_images": sum(count == 0 for count in box_counts),
            "single_box_images": sum(count == 1 for count in box_counts),
            "multi_box_images": sum(count >= 2 for count in box_counts),
            "total_boxes": sum(box_counts),
            "boxes_per_image": _quantiles([float(count) for count in box_counts]),
            "width": _quantiles(widths),
            "height": _quantiles(heights),
            "area": _quantiles(areas),
            "height_over_width": _quantiles(aspect_ratios),
        },
        "seed_labels": {
            "positive": seed_positive,
            "negative": seed_negative,
            "positive_fraction": (
                round(seed_positive / len(materialised), 6) if materialised else None
            ),
        },
        "attributes": {
            "dimension": len(attribute_names),
            "missing_by_name": missing_counts,
            "total_missing": total_missing,
            "missing_fraction": (
                round(total_missing / total_attribute_values, 6)
                if total_attribute_values
                else None
            ),
        },
        "subsets": dict(sorted(subset_counts.items())),
        "anchor_advice": _anchor_advice(areas, aspect_ratios),
    }
