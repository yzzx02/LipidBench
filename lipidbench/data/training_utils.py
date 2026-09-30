"""Batch helpers for the maintained candidate and peak detection trainer."""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import torch

from .peak_manifest import PeakManifestRecord

def balanced_candidate_subset(
    records: Iterable[PeakManifestRecord],
    limit: int,
    *,
    seed: int,
) -> list[PeakManifestRecord]:
    materialised = list(records)
    if limit <= 0 or limit >= len(materialised):
        return materialised

    rng = random.Random(seed)
    shuffled = materialised[:]
    rng.shuffle(shuffled)
    selected: list[PeakManifestRecord] = []
    selected_ids: set[str] = set()
    predicates = (
        lambda record: record.seed_label == 0,
        lambda record: record.seed_label == 1,
        lambda record: len(record.boxes) == 0,
        lambda record: len(record.boxes) == 1,
        lambda record: len(record.boxes) > 1,
    )
    for predicate in predicates:
        match = next(
            (
                record
                for record in shuffled
                if record.sample_id not in selected_ids and predicate(record)
            ),
            None,
        )
        if match is not None and len(selected) < limit:
            selected.append(match)
            selected_ids.add(match.sample_id)
    for record in shuffled:
        if len(selected) >= limit:
            break
        if record.sample_id not in selected_ids:
            selected.append(record)
            selected_ids.add(record.sample_id)
    return selected


def move_peak_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "images": [image.to(device) for image in batch["images"]],
        "targets": [
            {name: value.to(device) for name, value in target.items()}
            for target in batch["targets"]
        ],
        "seed_boxes": [box.to(device) for box in batch["seed_boxes"]],
        "attributes": batch["attributes"].to(device),
        "attribute_masks": batch["attribute_masks"].to(device),
        "seed_labels": batch["seed_labels"].to(device),
    }
