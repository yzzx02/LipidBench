from __future__ import annotations

import unittest

import numpy as np

from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries


def _gaussian(rt: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((rt - mu) / sigma) ** 2)


class RTBoundaryRefinerTest(unittest.TestCase):
    def test_keeps_local_apex_when_remote_peak_is_stronger(self) -> None:
        rt = np.linspace(4.0, 4.4, 401)
        y = _gaussian(rt, 4.10, 0.010, 120.0) + _gaussian(rt, 4.24, 0.012, 320.0)

        out = refine_peak_boundaries(
            rt,
            y,
            4.10,
            rtmin_hint=4.08,
            rtmax_hint=4.12,
            search_half_window_min=0.20,
        )

        self.assertEqual(out.status, "ok")
        self.assertLess(abs(out.apex_rt - 4.10), 0.02)
        self.assertTrue(out.rtmin <= 4.10 <= out.rtmax)

    def test_recenter_from_rt_when_old_bounds_are_shifted(self) -> None:
        rt = np.linspace(4.6, 5.3, 701)
        y = _gaussian(rt, 5.00, 0.018, 240.0)

        out = refine_peak_boundaries(
            rt,
            y,
            5.00,
            rtmin_hint=4.70,
            rtmax_hint=4.75,
            search_half_window_min=0.30,
        )

        self.assertEqual(out.status, "ok")
        self.assertFalse(out.old_rt_in_bounds)
        self.assertLess(abs(out.apex_rt - 5.00), 0.03)
        self.assertTrue(out.rtmin <= 5.00 <= out.rtmax)

    def test_handles_noisy_peak_without_exploding_width(self) -> None:
        rng = np.random.default_rng(123)
        rt = np.linspace(5.6, 6.4, 801)
        y = _gaussian(rt, 6.00, 0.020, 220.0)
        y += rng.normal(0.0, 12.0, size=rt.size)
        y += (rng.random(rt.size) < 0.02) * rng.uniform(10.0, 40.0, size=rt.size)
        y = np.clip(y, 0.0, None)

        out = refine_peak_boundaries(
            rt,
            y,
            6.00,
            rtmin_hint=5.95,
            rtmax_hint=6.05,
            search_half_window_min=0.20,
            local_half_window_min=0.40,
        )

        self.assertEqual(out.status, "ok")
        self.assertLess(abs(out.apex_rt - 6.00), 0.03)
        self.assertGreater(out.width_sec, 2.0)
        self.assertLess(out.width_sec, 60.0)
        self.assertTrue(out.rtmin <= out.apex_rt <= out.rtmax)

    def test_shrinks_overwide_boundary(self) -> None:
        rt = np.linspace(6.0, 8.0, 2001)
        y = _gaussian(rt, 7.00, 0.030, 260.0)
        y += np.where(rt > 7.03, 25.0 * np.exp(-(rt - 7.03) / 0.25), 0.0)

        out = refine_peak_boundaries(
            rt,
            y,
            7.00,
            rtmin_hint=6.95,
            rtmax_hint=7.05,
            search_half_window_min=0.30,
            max_expand_scans=300,
            max_expand_min=0.80,
            oversize_factor=1.10,
        )

        self.assertEqual(out.status, "ok")
        self.assertTrue(out.oversized_shrink)
        self.assertEqual(out.bound_mode, "core_shrink")
        self.assertLess(out.width_sec, 80.0)

    def test_returns_zero_apex_when_no_local_signal_exists(self) -> None:
        rt = np.linspace(3.8, 4.4, 601)
        y = _gaussian(rt, 4.40, 0.004, 300.0)

        out = refine_peak_boundaries(
            rt,
            y,
            4.10,
            rtmin_hint=4.08,
            rtmax_hint=4.12,
            search_half_window_min=0.25,
        )

        self.assertEqual(out.status, "zero_apex")
        self.assertAlmostEqual(out.rtmin, out.rtmax, places=9)
        self.assertLess(abs(out.apex_rt - 4.10), 0.01)


if __name__ == "__main__":
    unittest.main()
