from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BASE_ATTRIBUTE_NAMES = (
    "SNR",
    "CV",
    "GS",
    "TPAS",
    "H2B",
    "ZZ",
    "DZZ",
    "PCC",
    "SKEW",
    "DENT",
    "DM",
    "ENT",
    "JAG",
    "SYM",
    "MOD",
    "EDGE",
)


def _normalise_box(box: Sequence[Any], name: str) -> tuple[float, float, float, float]:
    if (
        isinstance(box, (str, bytes))
        or not isinstance(box, Sequence)
        or len(box) != 4
    ):
        raise ValueError(f"{name} must contain exactly four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} coordinates must be finite")
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0:
        raise ValueError(f"{name} coordinates must be non-negative")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must have positive width and height")
    return values


def _normalise_attribute(value: Any, index: int) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"attributes[{index}] must be finite or null")
    return numeric


@dataclass(frozen=True)
class PeakManifestRecord:
    """One EIC window and its model inputs, without opening the image.

    Attributes belong only to the original ``seed_box``. They must never be
    copied to full-window detection proposals or predicted instances.
    """

    sample_id: str
    image_path: str
    boxes: tuple[tuple[float, float, float, float], ...]
    seed_box: tuple[float, float, float, float]
    seed_label: int
    attributes: tuple[float | None, ...]
    source_file: str
    study_id: str
    instrument_id: str
    subsets: tuple[str, ...] = ()
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
    ) -> "PeakManifestRecord":
        if expected_attr_dim <= 0:
            raise ValueError("expected_attr_dim must be positive")

        required = {
            "sample_id",
            "image_path",
            "boxes",
            "seed_box",
            "seed_label",
            "attributes",
            "source_file",
            "study_id",
            "instrument_id",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"manifest record is missing required fields: {missing}")

        text_fields = {
            name: str(value[name]).strip()
            for name in ("sample_id", "image_path", "source_file", "study_id", "instrument_id")
        }
        empty = sorted(name for name, text in text_fields.items() if not text)
        if empty:
            raise ValueError(f"manifest text fields must be non-empty: {empty}")

        raw_boxes = value["boxes"]
        if isinstance(raw_boxes, (str, bytes)) or not isinstance(raw_boxes, Sequence):
            raise ValueError("boxes must be a sequence of [x1, y1, x2, y2] boxes")
        boxes = tuple(
            _normalise_box(box, f"boxes[{index}]") for index, box in enumerate(raw_boxes)
        )
        seed_box = _normalise_box(value["seed_box"], "seed_box")

        seed_label = value["seed_label"]
        if isinstance(seed_label, bool) or seed_label not in (0, 1, 0.0, 1.0):
            raise ValueError("seed_label must be binary 0 or 1")

        raw_attributes = value["attributes"]
        if isinstance(raw_attributes, (str, bytes)) or not isinstance(raw_attributes, Sequence):
            raise ValueError("attributes must be a sequence")
        if len(raw_attributes) != expected_attr_dim:
            raise ValueError(
                f"attributes must contain {expected_attr_dim} values, got {len(raw_attributes)}"
            )
        attributes = tuple(
            _normalise_attribute(attribute, index)
            for index, attribute in enumerate(raw_attributes)
        )

        raw_subsets = value.get("subsets", ())
        if isinstance(raw_subsets, (str, bytes)) or not isinstance(raw_subsets, Sequence):
            raise ValueError("subsets must be a sequence of strings")
        subsets = tuple(str(subset).strip() for subset in raw_subsets)
        if any(not subset for subset in subsets):
            raise ValueError("subsets cannot contain empty names")

        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("metadata must be a mapping")

        return cls(
            sample_id=text_fields["sample_id"],
            image_path=text_fields["image_path"],
            boxes=boxes,
            seed_box=seed_box,
            seed_label=int(seed_label),
            attributes=attributes,
            source_file=text_fields["source_file"],
            study_id=text_fields["study_id"],
            instrument_id=text_fields["instrument_id"],
            subsets=subsets,
            metadata=dict(raw_metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "boxes": [list(box) for box in self.boxes],
            "seed_box": list(self.seed_box),
            "seed_label": self.seed_label,
            "attributes": list(self.attributes),
            "source_file": self.source_file,
            "study_id": self.study_id,
            "instrument_id": self.instrument_id,
            "subsets": list(self.subsets),
            "metadata": dict(self.metadata),
        }


def validate_manifest_records(
    records: Iterable[PeakManifestRecord],
    *,
    expected_attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
) -> list[PeakManifestRecord]:
    if expected_attr_dim <= 0:
        raise ValueError("expected_attr_dim must be positive")
    materialised = list(records)
    sample_ids: set[str] = set()
    duplicates: set[str] = set()
    for record in materialised:
        if not isinstance(record, PeakManifestRecord):
            raise TypeError("records must contain PeakManifestRecord instances")
        if len(record.attributes) != expected_attr_dim:
            raise ValueError(
                f"sample {record.sample_id!r} has {len(record.attributes)} attributes; "
                f"expected {expected_attr_dim}"
            )
        if record.sample_id in sample_ids:
            duplicates.add(record.sample_id)
        sample_ids.add(record.sample_id)
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {sorted(duplicates)}")
    return materialised


def load_manifest_jsonl(
    path: str | Path,
    *,
    expected_attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
) -> list[PeakManifestRecord]:
    """Load a JSONL manifest only when explicitly called by a data workflow."""

    manifest_path = Path(path)
    records: list[PeakManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("record must be a JSON object")
                records.append(
                    PeakManifestRecord.from_mapping(
                        value,
                        expected_attr_dim=expected_attr_dim,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
    return validate_manifest_records(records, expected_attr_dim=expected_attr_dim)


def save_manifest_jsonl(
    records: Iterable[PeakManifestRecord],
    path: str | Path,
    *,
    expected_attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
) -> None:
    materialised = validate_manifest_records(
        records,
        expected_attr_dim=expected_attr_dim,
    )
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in materialised:
            handle.write(json.dumps(record.to_mapping(), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
