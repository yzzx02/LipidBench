from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from lipidbench.utils.peak_attributes import (
    ADDITIONAL_PEAK_ATTRIBUTE_COLUMNS,
    LITERATURE_TOP_COLUMNS,
    PEAK_ATTRIBUTE_COLUMNS,
    _compute_additional_peak_features,
    _compute_literature_top_features,
    _moving_average3,
)


def test_attribute_column_order_appends_three_without_reordering_original_thirteen() -> None:
    assert PEAK_ATTRIBUTE_COLUMNS[:13] == LITERATURE_TOP_COLUMNS
    assert PEAK_ATTRIBUTE_COLUMNS[13:] == ADDITIONAL_PEAK_ATTRIBUTE_COLUMNS
    assert PEAK_ATTRIBUTE_COLUMNS == [*LITERATURE_TOP_COLUMNS, "SYM", "MOD", "EDGE"]


def test_symmetric_profile_has_unit_symmetry_and_low_edges() -> None:
    x = np.asarray([0.0, 1.0, 4.0, 10.0, 4.0, 1.0, 0.0])
    attrs = _compute_additional_peak_features(x, apex_idx=3)
    assert attrs["SYM"] == pytest.approx(1.0)
    assert attrs["MOD"] == pytest.approx(0.0)
    assert attrs["EDGE"] == pytest.approx(0.1)


def test_asymmetric_profile_reduces_symmetry() -> None:
    symmetric = np.asarray([0.0, 1.0, 4.0, 10.0, 4.0, 1.0, 0.0])
    tailed = np.asarray([0.0, 1.0, 4.0, 10.0, 8.0, 7.0, 6.0])
    sym = _compute_additional_peak_features(symmetric, apex_idx=3)["SYM"]
    asym = _compute_additional_peak_features(tailed, apex_idx=3)["SYM"]
    assert 0.0 <= asym < sym <= 1.0


def test_mod_uses_secondary_prominence_and_excludes_apex_neighbourhood() -> None:
    x = np.asarray([0.0, 2.0, 12.0, 2.0, 0.0, 0.0, 8.0, 0.0, 0.0])
    apex_idx = 2
    y = _moving_average3(x)
    peaks, props = signal.find_peaks(y, prominence=0.0)
    keep = np.abs(peaks - apex_idx) > 2
    expected = np.clip(np.max(props["prominences"][keep]) / y[apex_idx], 0.0, 1.0)
    attrs = _compute_additional_peak_features(x, apex_idx=apex_idx)
    assert attrs["MOD"] == pytest.approx(expected)
    assert attrs["MOD"] > 0.0


def test_peak_inside_apex_exclusion_radius_does_not_raise_mod() -> None:
    x = np.asarray([0.0, 0.0, 10.0, 0.0, 7.0, 0.0, 0.0])
    attrs = _compute_additional_peak_features(x, apex_idx=2)
    assert attrs["MOD"] == pytest.approx(0.0)


def test_edge_ratio_is_continuous_and_not_clipped_to_one() -> None:
    x = np.asarray([20.0, 20.0, 20.0, 10.0, 20.0, 20.0, 20.0])
    attrs = _compute_additional_peak_features(x, apex_idx=3)
    assert attrs["EDGE"] == pytest.approx(2.0)


def test_original_thirteen_calculation_is_independent_of_extension() -> None:
    rt = np.arange(7, dtype=np.float64)
    x = np.asarray([0.0, 1.0, 4.0, 10.0, 4.0, 1.0, 0.0])
    before = _compute_literature_top_features(rt, x, apex_idx=3)
    _compute_additional_peak_features(x, apex_idx=3)
    after = _compute_literature_top_features(rt, x, apex_idx=3)
    for name in LITERATURE_TOP_COLUMNS:
        if np.isnan(before[name]):
            assert np.isnan(after[name])
        else:
            assert after[name] == pytest.approx(before[name], nan_ok=True)
