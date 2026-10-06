import numpy as np
import pytest
from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries, refine_peak_boundaries_guarded


def trace():
    rt = np.linspace(0, 2, 201)
    return rt, 10000 * np.exp(-((rt - 1) / .07) ** 2)


def test_guard_caps_extension_without_changing_core():
    rt, y = trace()
    core = refine_peak_boundaries(rt, y, 1, rtmin_hint=.96, rtmax_hint=1.04)
    guarded = refine_peak_boundaries_guarded(rt, y, 1, rtmin_hint=.96, rtmax_hint=1.04)
    assert guarded.rtmin == max(core.rtmin, .96 - np.median(np.diff(rt)))
    assert guarded.rtmax == min(core.rtmax, 1.04 + np.median(np.diff(rt)))
    assert guarded.refinement == core
    assert guarded.status == 'ok'
    assert guarded.rtmin <= guarded.apex_rt <= guarded.rtmax


def test_guard_marks_apex_excluded_instead_of_accepting_reference():
    rt, y = trace()
    result = refine_peak_boundaries_guarded(rt, y, 1, rtmin_hint=.1, rtmax_hint=.2)
    assert result.status in {'guard_no_overlap', 'guard_apex_outside'}


@pytest.mark.parametrize('options', [dict(rtmin_hint=1, rtmax_hint=0),
                                     dict(rtmin_hint=np.nan, rtmax_hint=2),
                                     dict(rtmin_hint=0, rtmax_hint=2, max_extension_scans=-1),
                                     dict(rtmin_hint=0, rtmax_hint=2, max_extension_scans=.5)])
def test_guard_rejects_invalid_configuration(options):
    rt, y = trace()
    with pytest.raises(ValueError):
        refine_peak_boundaries_guarded(rt, y, 1, **options)


def test_guard_handles_short_empty_trace():
    result = refine_peak_boundaries_guarded(np.array([]), np.array([]), 1, rtmin_hint=.9, rtmax_hint=1.1)
    assert result.status == 'empty_trace'
    assert not result.guard_applied
