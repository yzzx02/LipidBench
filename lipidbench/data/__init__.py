"""Manifest, Dataset, splitting, and audit interfaces for PeakTruthLab."""

from .attribute_preprocessing import AttributePreprocessor
from .peak_dataset import (
    PeakMultiTaskDataset,
    collate_peak_multitask_batch,
    load_rgb_image_tensor,
)
from .peak_manifest import (
    BASE_ATTRIBUTE_NAMES,
    PeakManifestRecord,
    load_manifest_jsonl,
    save_manifest_jsonl,
    validate_manifest_records,
)
from .peak_split_audit import (
    assert_source_file_disjoint,
    audit_manifest_records,
    split_grouped_records,
)

__all__ = [
    "AttributePreprocessor",
    "BASE_ATTRIBUTE_NAMES",
    "PeakManifestRecord",
    "PeakMultiTaskDataset",
    "assert_source_file_disjoint",
    "audit_manifest_records",
    "collate_peak_multitask_batch",
    "load_manifest_jsonl",
    "load_rgb_image_tensor",
    "save_manifest_jsonl",
    "split_grouped_records",
    "validate_manifest_records",
]
