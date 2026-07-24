from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .peak_manifest import BASE_ATTRIBUTE_NAMES, PeakManifestRecord, validate_manifest_records


ImageLoader = Callable[[Path], torch.Tensor]


def load_rgb_image_tensor(path: Path) -> torch.Tensor:
    """Load one RGB image as float32 ``[3,H,W]`` in the range [0, 1]."""

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return pil_to_tensor(rgb).to(dtype=torch.float32).div_(255.0)


def collate_peak_multitask_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve variable box counts while stacking fixed-size Seed inputs."""

    if not batch:
        raise ValueError("batch must contain at least one sample")
    return {
        "images": [sample["image"] for sample in batch],
        "targets": [sample["target"] for sample in batch],
        "seed_boxes": [sample["seed_box"] for sample in batch],
        "attributes": torch.stack([sample["attributes"] for sample in batch]),
        "attribute_masks": torch.stack([sample["attribute_mask"] for sample in batch]),
        "seed_labels": torch.stack([sample["seed_label"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


class PeakMultiTaskDataset(Dataset[dict[str, Any]]):
    """Convert validated manifest records into PeakMultiTaskRCNN inputs.

    Construction validates metadata only and does not open images. Image I/O
    occurs lazily in ``__getitem__`` through an injectable loader, which lets
    tests and future pipelines use synthetic tensors without real data access.
    Missing attributes are returned as NaN together with ``attribute_mask``;
    an explicit training-time imputation policy must be defined before these
    tensors can be passed to the model.
    """

    def __init__(
        self,
        records: Iterable[PeakManifestRecord],
        *,
        image_root: str | Path | None = None,
        image_loader: ImageLoader = load_rgb_image_tensor,
        attr_dim: int = len(BASE_ATTRIBUTE_NAMES),
    ) -> None:
        self.records = validate_manifest_records(records, expected_attr_dim=attr_dim)
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_loader = image_loader
        self.attr_dim = int(attr_dim)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if self.image_root is None:
            return path

        # Retain the source/sample suffix when a manifest was prepared on a
        # different Windows drive or operating system. This keeps immutable
        # manifests portable across Windows CUDA, Windows ROCm, WSL, and Linux.
        windows_path = PureWindowsPath(image_path)
        root_name = self.image_root.name.casefold()
        if windows_path.drive:
            matching_indices = [
                index
                for index, part in enumerate(windows_path.parts)
                if part.casefold() == root_name
            ]
            if not matching_indices:
                raise ValueError(
                    f"Windows image path {image_path!r} does not contain "
                    f"image_root directory {self.image_root.name!r}"
                )
            suffix = windows_path.parts[matching_indices[-1] + 1 :]
            return self.image_root.joinpath(*suffix)

        matching_indices = [
            index
            for index, part in enumerate(path.parts)
            if part.casefold() == root_name
        ]
        if matching_indices:
            suffix = path.parts[matching_indices[-1] + 1 :]
            return self.image_root.joinpath(*suffix)
        if not path.is_absolute():
            path = self.image_root / path
        return path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self.image_loader(self._resolve_image_path(record.image_path))
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
            raise ValueError(f"image_loader must return Tensor[3,H,W], got {shape}")
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("image_loader must return a finite floating-point tensor")

        height, width = image.shape[-2:]
        all_boxes = (*record.boxes, record.seed_box)
        if any(box[2] > width or box[3] > height for box in all_boxes):
            raise ValueError(
                f"sample {record.sample_id!r} contains a box outside "
                f"image bounds (height={height}, width={width})"
            )

        if record.boxes:
            boxes = torch.tensor(record.boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((len(record.boxes),), dtype=torch.int64)

        attribute_mask = torch.tensor(
            [value is not None for value in record.attributes],
            dtype=torch.bool,
        )
        attributes = torch.tensor(
            [float("nan") if value is None else value for value in record.attributes],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "target": {"boxes": boxes, "labels": labels},
            "seed_box": torch.tensor([record.seed_box], dtype=torch.float32),
            "seed_label": torch.tensor(float(record.seed_label), dtype=torch.float32),
            "attributes": attributes,
            "attribute_mask": attribute_mask,
            "metadata": {
                **dict(record.metadata),
                "sample_id": record.sample_id,
                "image_path": record.image_path,
                "source_file": record.source_file,
                "study_id": record.study_id,
                "instrument_id": record.instrument_id,
                "subsets": record.subsets,
            },
        }
