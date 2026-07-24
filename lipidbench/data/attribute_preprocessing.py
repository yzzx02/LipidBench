from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .peak_manifest import BASE_ATTRIBUTE_NAMES, PeakManifestRecord


@dataclass(frozen=True)
class AttributePreprocessor:
    """Train-only median imputation followed by z-score standardisation."""

    attribute_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    missing_counts: tuple[int, ...]
    fitted_samples: int
    near_zero_variance: tuple[str, ...] = ()

    @property
    def attr_dim(self) -> int:
        return len(self.attribute_names)

    @classmethod
    def fit(
        cls,
        records: Iterable[PeakManifestRecord],
        *,
        attribute_names: Sequence[str] = BASE_ATTRIBUTE_NAMES,
        minimum_scale: float = 1e-12,
    ) -> "AttributePreprocessor":
        materialised = list(records)
        names = tuple(str(name) for name in attribute_names)
        if not materialised:
            raise ValueError("cannot fit attribute preprocessing on an empty train set")
        if not names:
            raise ValueError("attribute_names must not be empty")
        if minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")

        columns: list[list[float]] = [[] for _ in names]
        missing_counts = [0 for _ in names]
        for record in materialised:
            if len(record.attributes) != len(names):
                raise ValueError(
                    f"sample {record.sample_id!r} has {len(record.attributes)} "
                    f"attributes; expected {len(names)}"
                )
            for index, value in enumerate(record.attributes):
                if value is None:
                    missing_counts[index] += 1
                else:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError(
                            f"sample {record.sample_id!r} attribute "
                            f"{names[index]!r} is not finite"
                        )
                    columns[index].append(numeric)

        empty = [names[index] for index, values in enumerate(columns) if not values]
        if empty:
            raise ValueError(
                "train set has attributes with no finite values; refusing to use "
                f"validation/test statistics: {empty}"
            )

        medians: list[float] = []
        means: list[float] = []
        scales: list[float] = []
        near_zero_variance: list[str] = []
        sample_count = len(materialised)
        for index, values in enumerate(columns):
            finite = torch.tensor(values, dtype=torch.float64)
            median = float(finite.median().item())
            completed = torch.full((sample_count,), median, dtype=torch.float64)
            cursor = 0
            for row_index, record in enumerate(materialised):
                value = record.attributes[index]
                if value is not None:
                    completed[row_index] = finite[cursor]
                    cursor += 1
            mean = float(completed.mean().item())
            scale = float(completed.std(unbiased=False).item())
            if not math.isfinite(scale) or scale < minimum_scale:
                scale = 1.0
                near_zero_variance.append(names[index])
            medians.append(median)
            means.append(mean)
            scales.append(scale)

        return cls(
            attribute_names=names,
            medians=tuple(medians),
            means=tuple(means),
            scales=tuple(scales),
            missing_counts=tuple(missing_counts),
            fitted_samples=sample_count,
            near_zero_variance=tuple(near_zero_variance),
        )

    def transform(
        self,
        attributes: torch.Tensor,
        attribute_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attributes.ndim != 2 or attributes.shape[1] != self.attr_dim:
            raise ValueError(
                f"attributes must have shape [B, {self.attr_dim}], "
                f"got {tuple(attributes.shape)}"
            )
        if not attributes.is_floating_point():
            raise ValueError("attributes must use a floating-point dtype")
        if attribute_mask is None:
            attribute_mask = torch.isfinite(attributes)
        if attribute_mask.shape != attributes.shape or attribute_mask.dtype != torch.bool:
            raise ValueError("attribute_mask must be bool with the same shape as attributes")
        if torch.any(attribute_mask & ~torch.isfinite(attributes)):
            raise ValueError("observed attributes marked by attribute_mask must be finite")

        medians = attributes.new_tensor(self.medians)
        means = attributes.new_tensor(self.means)
        scales = attributes.new_tensor(self.scales)
        imputed = torch.where(attribute_mask, attributes, medians)
        transformed = (imputed - means) / scales
        if not torch.isfinite(transformed).all():
            raise ValueError("attribute preprocessing produced NaN or Inf")
        return transformed

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fit_partition": "train",
            "fitted_samples": self.fitted_samples,
            "imputation": "per_attribute_train_median",
            "scaling": "per_attribute_train_mean_population_std_after_imputation",
            "preserve_missing_mask": True,
            "attribute_order": list(self.attribute_names),
            "statistics": {
                name: {
                    "median": self.medians[index],
                    "mean": self.means[index],
                    "scale": self.scales[index],
                    "train_missing": self.missing_counts[index],
                }
                for index, name in enumerate(self.attribute_names)
            },
            "near_zero_variance_scale_set_to_one": list(self.near_zero_variance),
            "data_leakage_guard": (
                "All statistics were fitted from the train manifest only; "
                "validation and test values were not inspected."
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttributePreprocessor":
        names = tuple(str(name) for name in value["attribute_order"])
        statistics = value["statistics"]
        if not isinstance(statistics, Mapping):
            raise ValueError("statistics must be a mapping")
        return cls(
            attribute_names=names,
            medians=tuple(float(statistics[name]["median"]) for name in names),
            means=tuple(float(statistics[name]["mean"]) for name in names),
            scales=tuple(float(statistics[name]["scale"]) for name in names),
            missing_counts=tuple(int(statistics[name]["train_missing"]) for name in names),
            fitted_samples=int(value["fitted_samples"]),
            near_zero_variance=tuple(
                str(name)
                for name in value.get("near_zero_variance_scale_set_to_one", ())
            ),
        )

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_mapping(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "AttributePreprocessor":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("attribute preprocessing JSON must contain an object")
        return cls.from_mapping(value)
