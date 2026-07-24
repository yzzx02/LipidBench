from __future__ import annotations

import torch

from lipidbench.data import AttributePreprocessor, PeakManifestRecord


def _record(sample_id: str, attributes: list[float | None]) -> PeakManifestRecord:
    return PeakManifestRecord.from_mapping(
        {
            "sample_id": sample_id,
            "image_path": f"{sample_id}.png",
            "boxes": [],
            "seed_box": [1, 1, 2, 2],
            "seed_label": 0,
            "attributes": attributes,
            "source_file": sample_id,
            "study_id": "study",
            "instrument_id": "instrument",
        },
        expected_attr_dim=len(attributes),
    )


def test_fit_imputes_train_median_and_standardises() -> None:
    records = [
        _record("a", [1.0, 10.0]),
        _record("b", [3.0, None]),
        _record("c", [100.0, 30.0]),
    ]
    preprocessor = AttributePreprocessor.fit(
        records,
        attribute_names=("first", "second"),
    )

    assert preprocessor.medians == (3.0, 10.0)
    assert preprocessor.missing_counts == (0, 1)
    raw = torch.tensor([[float("nan"), 30.0]], dtype=torch.float32)
    mask = torch.tensor([[False, True]])
    transformed = preprocessor.transform(raw, mask)
    expected = torch.tensor(
        [
            [
                (3.0 - preprocessor.means[0]) / preprocessor.scales[0],
                (30.0 - preprocessor.means[1]) / preprocessor.scales[1],
            ]
        ]
    )
    torch.testing.assert_close(transformed, expected)
    assert torch.isfinite(transformed).all()


def test_fit_rejects_all_missing_train_attribute() -> None:
    records = [
        _record("a", [1.0, None]),
        _record("b", [2.0, None]),
    ]
    try:
        AttributePreprocessor.fit(
            records,
            attribute_names=("first", "second"),
        )
    except ValueError as exc:
        assert "validation/test" in str(exc)
        assert "second" in str(exc)
    else:
        raise AssertionError("all-missing train attribute must be rejected")
