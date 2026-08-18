from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, ScalarFormatter


FONT_FAMILIES = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILIES,
        "axes.unicode_minus": False,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lipidbench.utils.peak_attributes import (  # noqa: E402
    PEAK_ATTRIBUTE_COLUMNS,
    _compute_one_feature_attributes,
    _extract_trace,
    load_ms1_spectra,
)


VALID_LABELS = {"True_Peak", "OUT_FIG"}
SELECTED_SOURCE_FILES = [
    "0001_MAY_RoCI-StEM_CP-287.mzML",
    "0021_MAY_ROCI-StEM_HN_575.mzML",
    "060-0145-006_018.mzML",
    "20180321_S00033936_P.mzML",
    "AG-88-11_r12-.mzML",
    "D03P_POS.mzML",
    "NIST_Full scan_1_POS.mzML",
    "frag1_pos20_1.mzML",
]

# The tolerance comparison is intentionally kept out of the second-round
# review images.  These source-level values preserve the best-supported
# generation setting from the first diagnostic pass; they can be revisited as
# a separate task without changing the annotation geometry review.
SOURCE_TOLERANCE_PPM = {
    "0001_MAY_RoCI-StEM_CP-287.mzML": 15.0,
    "0021_MAY_ROCI-StEM_HN_575.mzML": 15.0,
    "060-0145-006_018.mzML": 10.0,
    "20180321_S00033936_P.mzML": 10.0,
    "AG-88-11_r12-.mzML": 10.0,
    "D03P_POS.mzML": 15.0,
    "NIST_Full scan_1_POS.mzML": 15.0,
    "frag1_pos20_1.mzML": 15.0,
}


@dataclass
class AxisSnapshot:
    fig: Any
    ax: Any
    image_rgb: np.ndarray
    plot_left: float
    plot_right: float
    plot_top: float
    plot_bottom: float


@dataclass
class AnalyticAxisTransform:
    center_rt: float
    plot_left: float
    plot_right: float
    plot_top: float
    plot_bottom: float
    y_bottom: float
    y_top: float
    image_height: float


def _tick_formatter(value: float, _pos: Any) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _extract_window(
    rt: np.ndarray,
    eic: np.ndarray,
    center_rt: float,
    window_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = float(window_min) / 2.0
    mask = (rt >= center_rt - half) & (rt <= center_rt + half)
    if np.any(mask):
        return rt[mask], eic[mask]
    return rt, eic


def _build_axis_snapshot(
    rt: np.ndarray,
    eic: np.ndarray,
    center_rt: float,
    width_px: int,
    height_px: int,
    dpi: int,
) -> AxisSnapshot:
    rt_win, eic_win = _extract_window(rt, eic, center_rt=center_rt, window_min=2.0)
    fig = plt.figure(
        figsize=(float(width_px) / float(dpi), float(height_px) / float(dpi)),
        dpi=int(dpi),
    )
    ax = fig.add_subplot(111)
    y = np.asarray(eic_win, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    ax.plot(rt_win, y, color="royalblue", linewidth=0.8)
    ax.set_xlim(center_rt - 1.0, center_rt + 1.0)
    y_max = float(np.max(y)) if y.size else 0.0
    y_top = max(y_max * 1.15, 0.5)
    y_bottom = -0.08 * y_top
    ax.set_ylim(y_bottom, y_top)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_tick_formatter))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=6)
    ax.yaxis.get_offset_text().set_size(6)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
    bbox = ax.bbox
    return AxisSnapshot(
        fig=fig,
        ax=ax,
        image_rgb=rgb,
        plot_left=float(bbox.x0),
        plot_right=float(bbox.x1),
        plot_top=float(height_px - bbox.y1),
        plot_bottom=float(height_px - bbox.y0),
    )


def _pixel_x_to_rt(ax: Any, x_px: float, image_height: float) -> float:
    if isinstance(ax, AnalyticAxisTransform):
        fraction = (
            float(x_px) - float(ax.plot_left)
        ) / max(float(ax.plot_right) - float(ax.plot_left), 1e-12)
        return float(ax.center_rt - 1.0 + 2.0 * fraction)
    x_data, _ = ax.transData.inverted().transform((float(x_px), float(image_height) / 2.0))
    return float(x_data)


def _rt_to_pixel_x(ax: Any, rt_value: float) -> float:
    if isinstance(ax, AnalyticAxisTransform):
        fraction = (
            float(rt_value) - (float(ax.center_rt) - 1.0)
        ) / 2.0
        return float(
            float(ax.plot_left)
            + fraction * (float(ax.plot_right) - float(ax.plot_left))
        )
    x_disp, _ = ax.transData.transform((float(rt_value), 0.0))
    return float(x_disp)


def _intensity_to_pixel_y(ax: Any, intensity: float, image_height: float) -> float:
    if isinstance(ax, AnalyticAxisTransform):
        fraction_from_top = (
            float(ax.y_top) - float(intensity)
        ) / max(float(ax.y_top) - float(ax.y_bottom), 1e-12)
        return float(
            float(ax.plot_top)
            + fraction_from_top
            * (float(ax.plot_bottom) - float(ax.plot_top))
        )
    _, y_disp = ax.transData.transform((0.0, float(intensity)))
    return float(image_height - y_disp)


def _curve_mask(image_rgb: np.ndarray) -> np.ndarray:
    target = np.asarray([65.0, 105.0, 225.0], dtype=np.float64)
    dist = np.sqrt(np.sum((image_rgb.astype(np.float64) - target) ** 2, axis=2))
    return dist <= 90.0


def _curve_fit_score(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
    ref = _curve_mask(reference_rgb)
    cand = _curve_mask(candidate_rgb)
    return _curve_mask_fit_score(ref, cand)


def _curve_mask_fit_score(ref: np.ndarray, cand: np.ndarray) -> float:
    if not np.any(ref) or not np.any(cand):
        return 0.0
    ref_d = binary_dilation(ref, iterations=1)
    cand_d = binary_dilation(cand, iterations=1)
    precision = float(np.sum(cand & ref_d)) / float(max(1, np.sum(cand)))
    recall = float(np.sum(ref & cand_d)) / float(max(1, np.sum(ref)))
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _detect_reference_plot_bounds(
    reference_rgb: np.ndarray,
) -> tuple[float, float, float, float]:
    """Detect the long black plot spines in the exact LabelMe reference image."""

    height, width = reference_rgb.shape[:2]
    dark = (
        (reference_rgb[:, :, 0] < 100)
        & (reference_rgb[:, :, 1] < 100)
        & (reference_rgb[:, :, 2] < 100)
    )
    y0 = max(0, int(round(height * 0.04)))
    y1 = min(height, int(round(height * 0.96)))
    x0 = max(0, int(round(width * 0.04)))
    x1 = min(width, int(round(width * 0.96)))
    column_counts = dark[y0:y1, :].sum(axis=0)
    row_counts = dark[:, x0:x1].sum(axis=1)
    x_candidates = np.flatnonzero(column_counts >= max(20, int((y1 - y0) * 0.60)))
    y_candidates = np.flatnonzero(row_counts >= max(20, int((x1 - x0) * 0.60)))

    def runs(values: np.ndarray) -> list[list[int]]:
        result: list[list[int]] = []
        for raw in values:
            value = int(raw)
            if not result or value > result[-1][-1] + 1:
                result.append([value])
            else:
                result[-1].append(value)
        return result

    x_runs = runs(x_candidates)
    y_runs = runs(y_candidates)
    if len(x_runs) < 2 or len(y_runs) < 2:
        raise RuntimeError("无法从LabelMe原图检测实际绘图区黑色边框")
    left = float(np.mean(x_runs[0]))
    right = float(np.mean(x_runs[-1]))
    top = float(np.mean(y_runs[0]))
    bottom = float(np.mean(y_runs[-1]))
    if right - left < width * 0.60 or bottom - top < height * 0.60:
        raise RuntimeError(
            f"检测到的绘图区过小：[{left}, {top}, {right}, {bottom}]"
        )
    return left, right, top, bottom


def _build_analytic_axis_snapshot(
    rt: np.ndarray,
    eic: np.ndarray,
    center_rt: float,
    reference_rgb: np.ndarray,
    plot_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[AxisSnapshot, np.ndarray]:
    height, width = reference_rgb.shape[:2]
    if plot_bounds is None:
        left, right, top, bottom = _detect_reference_plot_bounds(reference_rgb)
    else:
        left, top, right, bottom = (float(value) for value in plot_bounds)
    rt_win, eic_win = _extract_window(
        rt, eic, center_rt=float(center_rt), window_min=2.0
    )
    y = np.asarray(eic_win, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    y_max = float(np.max(y)) if y.size else 0.0
    y_top = max(y_max * 1.15, 0.5)
    y_bottom = -0.08 * y_top
    transform = AnalyticAxisTransform(
        center_rt=float(center_rt),
        plot_left=left,
        plot_right=right,
        plot_top=top,
        plot_bottom=bottom,
        y_bottom=y_bottom,
        y_top=y_top,
        image_height=float(height),
    )

    canvas = Image.new("1", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(canvas)
    points: list[tuple[float, float]] = []
    for rt_value, intensity in zip(rt_win, y):
        x_value = _rt_to_pixel_x(transform, float(rt_value))
        y_value = _intensity_to_pixel_y(
            transform, float(intensity), float(height)
        )
        points.append((x_value, y_value))
    if len(points) >= 2:
        draw.line(points, fill=1, width=2)
    elif len(points) == 1:
        x_value, y_value = points[0]
        draw.point((x_value, y_value), fill=1)
    curve_mask = np.asarray(canvas, dtype=bool)
    snapshot = AxisSnapshot(
        fig=None,
        ax=transform,
        image_rgb=reference_rgb,
        plot_left=left,
        plot_right=right,
        plot_top=top,
        plot_bottom=bottom,
    )
    return snapshot, curve_mask


def _recover_original_axis_center_analytic(
    rt: np.ndarray,
    eic: np.ndarray,
    current_seed_rt: float,
    reference_rgb: np.ndarray,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[AxisSnapshot, float, str, float]:
    """Recover center and transform without rendering a Matplotlib figure."""

    del width_px, height_px, dpi
    seed_snapshot, seed_mask = _build_analytic_axis_snapshot(
        rt, eic, float(current_seed_rt), reference_rgb
    )
    ref_mask = _curve_mask(reference_rgb)
    ref_dilated = binary_dilation(ref_mask, iterations=1).astype(np.float64)
    candidate = seed_mask.astype(np.float64)
    correlation = fftconvolve(
        ref_dilated, candidate[:, ::-1], mode="full", axes=1
    ).sum(axis=0)
    peak_position = int(np.argmax(correlation))
    lag_pixels = float(peak_position - (candidate.shape[1] - 1))
    if 0 < peak_position < len(correlation) - 1:
        left_score = float(correlation[peak_position - 1])
        middle_score = float(correlation[peak_position])
        right_score = float(correlation[peak_position + 1])
        denominator = left_score - 2.0 * middle_score + right_score
        if abs(denominator) > 1e-12:
            lag_pixels += float(
                np.clip(
                    0.5 * (left_score - right_score) / denominator,
                    -0.5,
                    0.5,
                )
            )
    pixels_per_minute = (
        float(seed_snapshot.plot_right) - float(seed_snapshot.plot_left)
    ) / 2.0
    estimated_center = float(current_seed_rt) - lag_pixels / max(
        pixels_per_minute, 1e-12
    )
    estimated_center = float(
        np.clip(
            estimated_center,
            float(current_seed_rt) - 0.40,
            float(current_seed_rt) + 0.40,
        )
    )
    if abs(estimated_center - float(current_seed_rt)) <= 0.0001:
        best_snapshot = seed_snapshot
        best_mask = seed_mask
        best_center = float(current_seed_rt)
    else:
        estimated_snapshot, estimated_mask = _build_analytic_axis_snapshot(
            rt, eic, estimated_center, reference_rgb
        )
        seed_score = _curve_mask_fit_score(ref_mask, seed_mask)
        estimated_score = _curve_mask_fit_score(ref_mask, estimated_mask)
        if estimated_score >= seed_score:
            best_snapshot = estimated_snapshot
            best_mask = estimated_mask
            best_center = estimated_center
        else:
            best_snapshot = seed_snapshot
            best_mask = seed_mask
            best_center = float(current_seed_rt)
    best_score = _curve_mask_fit_score(ref_mask, best_mask)
    source = (
        f"由LabelMe原图蓝线解析相关恢复绘图中心RT={best_center:.6f}"
        if abs(best_center - float(current_seed_rt)) > 0.002
        else "当前Seed RT近似回溯LabelMe原图（解析相关）"
    )
    return best_snapshot, best_center, source, best_score


def _load_labelme_reference_image(
    obj: dict[str, Any],
    external_image_path: Path,
) -> tuple[np.ndarray, str, bool]:
    """Return the exact image on which LabelMe coordinates were authored.

    LabelMe normally displays ``imageData`` when it is embedded in the JSON.
    The external PNG is only authoritative when the JSON has no embedded image.
    ``pixels_match_external`` is recorded separately so stale/regenerated PNGs
    cannot silently corrupt the coordinate mapping again.
    """
    external_rgb = np.asarray(Image.open(external_image_path).convert("RGB"))
    image_data = obj.get("imageData")
    if not image_data:
        return external_rgb, "外部PNG（JSON无内嵌imageData）", True
    raw = base64.b64decode(str(image_data))
    embedded_rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    if embedded_rgb.shape != external_rgb.shape:
        return embedded_rgb, "LabelMe JSON内嵌原图（与外部PNG尺寸不同）", False
    pixels_match = bool(np.array_equal(embedded_rgb, external_rgb))
    source = (
        "LabelMe JSON内嵌原图（与外部PNG像素一致）"
        if pixels_match
        else "LabelMe JSON内嵌原图（外部PNG为不同版本）"
    )
    return embedded_rgb, source, pixels_match


def _recover_original_axis_center(
    rt: np.ndarray,
    eic: np.ndarray,
    current_seed_rt: float,
    reference_rgb: np.ndarray,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[AxisSnapshot, float, str, float]:
    """Recover the x-axis center actually used by the annotation image."""

    def score_at(center: float) -> float:
        candidate = _build_axis_snapshot(rt, eic, center, width_px, height_px, dpi)
        try:
            return _curve_fit_score(reference_rgb, candidate.image_rgb)
        finally:
            plt.close(candidate.fig)

    seed_score = score_at(float(current_seed_rt))
    if seed_score >= 0.995:
        snapshot = _build_axis_snapshot(rt, eic, current_seed_rt, width_px, height_px, dpi)
        return snapshot, float(current_seed_rt), "当前Seed RT与LabelMe原图精确回溯", float(seed_score)

    coarse_centers = np.arange(current_seed_rt - 0.40, current_seed_rt + 0.4001, 0.02)
    coarse_scores = [(float(center), score_at(float(center))) for center in coarse_centers]
    coarse_best_center, _ = max(coarse_scores, key=lambda pair: pair[1])
    fine_centers = np.arange(coarse_best_center - 0.03, coarse_best_center + 0.0301, 0.001)
    fine_scores = [(float(center), score_at(float(center))) for center in fine_centers]
    best_center, best_score = max(coarse_scores + fine_scores, key=lambda pair: pair[1])
    snapshot = _build_axis_snapshot(rt, eic, best_center, width_px, height_px, dpi)
    source = (
        f"由LabelMe内嵌原图蓝线拟合恢复绘图中心RT={best_center:.6f}"
        if abs(best_center - current_seed_rt) > 0.002
        else "当前Seed RT近似回溯LabelMe原图"
    )
    return snapshot, float(best_center), source, float(best_score)


def _recover_original_axis_center_fast(
    rt: np.ndarray,
    eic: np.ndarray,
    current_seed_rt: float,
    reference_rgb: np.ndarray,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[AxisSnapshot, float, str, float]:
    """Recover the image center with x-only image correlation and local refinement.

    The original exhaustive search renders roughly one hundred candidate
    figures per image.  Here the current-Seed rendering is aligned to the
    LabelMe blue curve by an x-only FFT correlation, then only a small local
    candidate set is rendered.  Very poor fits still fall back to the
    exhaustive routine so low-quality images are not silently accepted.
    """

    seed_snapshot = _build_axis_snapshot(
        rt, eic, float(current_seed_rt), width_px, height_px, dpi
    )
    seed_score = _curve_fit_score(reference_rgb, seed_snapshot.image_rgb)
    if seed_score >= 0.995:
        return (
            seed_snapshot,
            float(current_seed_rt),
            "当前Seed RT与LabelMe原图精确回溯（快速相关）",
            float(seed_score),
        )

    ref_mask = binary_dilation(_curve_mask(reference_rgb), iterations=1).astype(
        np.float64
    )
    candidate_mask = _curve_mask(seed_snapshot.image_rgb).astype(np.float64)
    if not np.any(ref_mask) or not np.any(candidate_mask):
        plt.close(seed_snapshot.fig)
        return _recover_original_axis_center(
            rt,
            eic,
            current_seed_rt,
            reference_rgb,
            width_px,
            height_px,
            dpi,
        )

    correlation = fftconvolve(
        ref_mask,
        candidate_mask[:, ::-1],
        mode="full",
        axes=1,
    ).sum(axis=0)
    peak_position = int(np.argmax(correlation))
    lag_pixels = float(peak_position - (candidate_mask.shape[1] - 1))
    if 0 < peak_position < len(correlation) - 1:
        left = float(correlation[peak_position - 1])
        middle = float(correlation[peak_position])
        right = float(correlation[peak_position + 1])
        denominator = left - 2.0 * middle + right
        if abs(denominator) > 1e-12:
            fractional = 0.5 * (left - right) / denominator
            lag_pixels += float(np.clip(fractional, -0.5, 0.5))

    pixels_per_minute = (
        float(seed_snapshot.plot_right) - float(seed_snapshot.plot_left)
    ) / 2.0
    estimated_center = float(current_seed_rt) - lag_pixels / max(
        pixels_per_minute, 1e-12
    )
    estimated_center = float(
        np.clip(
            estimated_center,
            float(current_seed_rt) - 0.40,
            float(current_seed_rt) + 0.40,
        )
    )

    if abs(estimated_center - float(current_seed_rt)) <= 0.0001:
        return (
            seed_snapshot,
            float(current_seed_rt),
            "当前Seed RT近似回溯LabelMe原图（快速相关）",
            float(seed_score),
        )
    estimated_snapshot = _build_axis_snapshot(
        rt, eic, estimated_center, width_px, height_px, dpi
    )
    estimated_score = _curve_fit_score(
        reference_rgb, estimated_snapshot.image_rgb
    )
    if seed_score >= estimated_score:
        plt.close(estimated_snapshot.fig)
        best_snapshot = seed_snapshot
        best_center = float(current_seed_rt)
        best_score = float(seed_score)
    else:
        plt.close(seed_snapshot.fig)
        best_snapshot = estimated_snapshot
        best_center = float(estimated_center)
        best_score = float(estimated_score)

    if best_score < 0.25:
        plt.close(best_snapshot.fig)
        return _recover_original_axis_center(
            rt,
            eic,
            current_seed_rt,
            reference_rgb,
            width_px,
            height_px,
            dpi,
        )

    source = (
        f"由LabelMe原图蓝线快速相关恢复绘图中心RT={best_center:.6f}"
        if abs(best_center - float(current_seed_rt)) > 0.002
        else "当前Seed RT近似回溯LabelMe原图（快速相关）"
    )
    return best_snapshot, float(best_center), source, float(best_score)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _find_mzml(source_path: str, source_file: str) -> Path | None:
    direct = Path(source_path)
    if direct.exists():
        return direct.resolve()
    search_root = REPO_ROOT / "data"
    matches = list(search_root.rglob(source_file)) if search_root.exists() else []
    return matches[0].resolve() if matches else None


def _valid_shapes(obj: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    shapes = obj.get("shapes")
    if not isinstance(shapes, list):
        return []
    out: list[tuple[int, dict[str, Any]]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        if str(shape.get("label", "")).strip() not in VALID_LABELS:
            continue
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        if not all(isinstance(p, list) and len(p) >= 2 for p in points[:2]):
            continue
        out.append((index, shape))
    return out


def _normalize_box(shape: dict[str, Any]) -> tuple[float, float, float, float]:
    p1, p2 = shape["points"][:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _smooth_trace(eic: np.ndarray) -> np.ndarray:
    """Light linearly weighted smoothing used only for boundary shape checks."""
    y = np.asarray(eic, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    if y.size < 5:
        return y
    weights = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    padded = np.pad(y, (2, 2), mode="edge")
    return np.convolve(padded, weights / weights.sum(), mode="valid")


def _nearest_idx(rt: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(rt - float(value)))) if rt.size else -1


def _refine_shape_boundary(
    rt: np.ndarray,
    eic: np.ndarray,
    original_rt: float,
    apex_idx: int,
    side: str,
    max_move_sec: float,
) -> tuple[float, bool, str]:
    """Conservatively move an edge to a nearby shape-confirmed tail minimum.

    A relative-height threshold is only a low-tail eligibility gate.  The
    endpoint itself is selected from the lightly smoothed slope direction,
    local minimum, and outward recovery.  This permits either a small inward
    shrink or outward extension while keeping the manual edge as the anchor.
    """
    if rt.size < 5 or apex_idx < 0:
        return float(original_rt), False, "扫描点不足"
    original_idx = _nearest_idx(rt, original_rt)
    if original_idx < 0:
        return float(original_rt), False, "无法定位原边界"
    if side == "left" and original_idx >= apex_idx:
        return float(original_rt), False, "左边界已接近或越过峰顶"
    if side == "right" and original_idx <= apex_idx:
        return float(original_rt), False, "右边界已接近或越过峰顶"

    ys = _smooth_trace(eic)
    plot_mask = np.abs(rt - rt[apex_idx]) <= 1.0
    local = ys[plot_mask] if np.any(plot_mask) else ys
    baseline = float(np.quantile(local, 0.20)) if local.size else 0.0
    low_part = local[local <= np.median(local)] if local.size else local
    sigma = (
        float(1.4826 * np.median(np.abs(low_part - np.median(low_part))))
        if low_part.size
        else 0.0
    )
    apex_value = float(max(ys[apex_idx], 0.0))
    # One percent is only the minimum low-tail eligibility gate.  Noise can
    # widen it up to five percent; neither number is used as the endpoint.
    signal_height = max(apex_value - baseline, 1e-12)
    low_fraction = min(0.05, max(0.01, 4.0 * sigma / signal_height))
    low_threshold = baseline + low_fraction * signal_height
    slope_noise = max(0.25 * sigma, 0.00025 * apex_value, 1e-12)
    current_value = float(ys[original_idx])
    current_z = max(current_value - baseline, 0.0) / signal_height
    max_move_min = float(max_move_sec) / 60.0
    outward_step = -1 if side == "left" else 1
    outward_neighbor = original_idx + outward_step
    inward_neighbor = original_idx - outward_step
    if not (0 <= outward_neighbor < len(ys) and 0 <= inward_neighbor < len(ys)):
        return float(original_rt), False, "原边界靠近数据端点，保持"

    outward_slope = float(ys[outward_neighbor] - ys[inward_neighbor])
    if outward_slope < -slope_noise:
        preferred_direction = "outward"
    elif outward_slope > slope_noise:
        preferred_direction = "inward"
    else:
        preferred_direction = "flat"

    def outward_order(start: int, stop: int) -> list[int]:
        step = outward_step if (stop - start) * outward_step >= 0 else -outward_step
        return list(range(start, stop + step, step))

    candidates: list[tuple[int, str, float]] = []
    for idx in range(1, len(ys) - 1):
        candidate_rt = float(rt[idx])
        if abs(candidate_rt - float(original_rt)) > max_move_min + 1e-12:
            continue
        if side == "left" and idx >= apex_idx:
            continue
        if side == "right" and idx <= apex_idx:
            continue
        is_minimum = bool(ys[idx] <= ys[idx - 1] and ys[idx] <= ys[idx + 1])
        if not is_minimum or float(ys[idx]) > low_threshold:
            continue

        # The final part of the peak must mostly descend towards the candidate.
        apex_to_candidate = outward_order(apex_idx, idx)
        tail = apex_to_candidate[-min(8, len(apex_to_candidate)) :]
        tail_delta = np.diff(ys[tail])
        tail_descent_fraction = (
            float(np.mean(tail_delta <= slope_noise)) if tail_delta.size else 1.0
        )
        if tail_descent_fraction < 0.70:
            continue

        # Require a real turn/plateau after the candidate, not an isolated
        # one-scan noise dip.  Two outward scans are used when available.
        recovery_indices = [
            idx + outward_step * distance
            for distance in (1, 2)
            if 0 <= idx + outward_step * distance < len(ys)
        ]
        if not recovery_indices:
            continue
        recovery = ys[recovery_indices]
        if not bool(np.all(recovery >= float(ys[idx]) - slope_noise)):
            continue

        direction = (
            "outward"
            if (candidate_rt - float(original_rt)) * outward_step > 0
            else "inward"
        )
        segment = outward_order(original_idx, idx)
        segment_delta = np.diff(ys[segment])
        if direction == "outward":
            shape_fraction = (
                float(np.mean(segment_delta <= slope_noise)) if segment_delta.size else 1.0
            )
        else:
            # Read from the candidate back out to the current manual edge.
            candidate_to_original = outward_order(idx, original_idx)
            outside_delta = np.diff(ys[candidate_to_original])
            shape_fraction = (
                float(np.mean(outside_delta >= -slope_noise)) if outside_delta.size else 1.0
            )
        required_shape_fraction = 0.80 if direction == "outward" else 0.75
        if shape_fraction < required_shape_fraction:
            continue
        improvement = current_value - float(ys[idx])
        improvement_z = improvement / signal_height
        if direction == "outward":
            # Expansion is allowed only when the manual edge is already on a
            # low tail, continues down by a meaningful relative amount, and
            # the next two scans confirm a turn rather than a noise chase.
            if current_z > min(0.03, low_fraction + 0.01):
                continue
            if improvement_z < max(slope_noise / signal_height, 0.08 * max(current_z, 1e-12)):
                continue
            if float(np.max(recovery) - ys[idx]) < slope_noise:
                continue
        elif improvement <= slope_noise:
            continue
        candidates.append((int(idx), direction, shape_fraction))

    if not candidates:
        return float(original_rt), False, "3秒内无经平滑、尾部单调性和回升共同确认的低点，保持原边界"

    preferred = [value for value in candidates if value[1] == preferred_direction]
    if preferred_direction == "flat":
        preferred = candidates
    if not preferred:
        return float(original_rt), False, "原边界附近斜率方向与候选低点不一致，保守保持"
    best_idx, direction, _ = min(
        preferred,
        key=lambda value: (
            abs(float(rt[value[0]]) - float(original_rt)),
            float(ys[value[0]]),
        ),
    )
    candidate_rt = float(rt[best_idx])
    median_spacing = float(np.median(np.diff(rt))) if rt.size >= 2 else 0.0
    if abs(candidate_rt - float(original_rt)) <= max(0.25 * median_spacing, 1e-12):
        return float(original_rt), False, "原人工边界已在形状确认的低点附近，保持"
    action = "向外小幅扩展" if direction == "outward" else "向峰顶小幅收缩"
    return candidate_rt, True, f"线性加权平滑+尾部单调性：{action}到首个可信局部低点"


def _shoulder_evidence_status(
    seed_diagnostics: dict[str, Any],
    height_ratio: float,
) -> str:
    """Return pass/uncertain/fail for a Seed hidden in a composite box."""
    if int(seed_diagnostics.get("point_count", 0)) < 3 or not np.isfinite(height_ratio):
        return "fail"
    if height_ratio >= 0.05:
        return "pass"
    if not bool(seed_diagnostics.get("max_at_edge", True)) and height_ratio >= 0.02:
        return "pass"
    if height_ratio >= 0.02:
        return "uncertain"
    return "fail"


def _logic_self_checks() -> dict[str, bool]:
    return {
        "明显平坦截断Seed不当肩峰": _shoulder_evidence_status(
            {"point_count": 20, "max_at_edge": True}, 0.01
        ) == "fail",
        "明确肩峰复合框判为通过": _shoulder_evidence_status(
            {"point_count": 20, "max_at_edge": False}, 0.08
        ) == "pass",
        "弱肩峰进入复核而非静默合并": _shoulder_evidence_status(
            {"point_count": 20, "max_at_edge": True}, 0.03
        ) == "uncertain",
    }


def _seed_eic_diagnostics(
    rt: np.ndarray,
    eic: np.ndarray,
    seed_rt: float,
    seed_rtmin: float,
    seed_rtmax: float,
) -> dict[str, Any]:
    """Reconstruct simple Seed geometry without replacing its stored 16 attributes."""
    mask = (rt >= seed_rtmin) & (rt <= seed_rtmax)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return {
            "width_sec": max(0.0, (seed_rtmax - seed_rtmin) * 60.0),
            "height": float("nan"),
            "height_rt": float("nan"),
            "area": float("nan"),
            "seed_intensity": float("nan"),
            "max_at_edge": True,
            "point_count": 0,
        }
    x = np.asarray(rt[indices], dtype=np.float64)
    y = np.asarray(eic[indices], dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    local_max_pos = int(np.argmax(y))
    max_index = int(indices[local_max_pos])
    seed_index = _nearest_idx(rt, seed_rt)
    plot_mask = np.abs(rt - seed_rt) <= 1.0
    baseline_source = eic[plot_mask] if np.any(plot_mask) else eic
    baseline_source = np.asarray(baseline_source, dtype=np.float64)
    baseline_source = baseline_source[np.isfinite(baseline_source)]
    baseline = float(np.quantile(np.maximum(baseline_source, 0.0), 0.20)) if baseline_source.size else 0.0
    net = np.maximum(y - baseline, 0.0)
    area = float(np.trapezoid(net, x * 60.0)) if x.size >= 2 else 0.0
    edge_margin = min(2, max(0, len(y) // 5))
    max_at_edge = bool(
        local_max_pos <= edge_margin
        or local_max_pos >= len(y) - 1 - edge_margin
    )
    return {
        "width_sec": max(0.0, (seed_rtmax - seed_rtmin) * 60.0),
        "height": float(y[local_max_pos]),
        "height_rt": float(rt[max_index]),
        "area": area,
        "seed_intensity": float(max(eic[seed_index], 0.0)) if seed_index >= 0 else float("nan"),
        "max_at_edge": max_at_edge,
        "point_count": int(indices.size),
    }


def _shared_valley_overrides(
    peak_infos: list[dict[str, Any]],
    rt: np.ndarray,
    eic: np.ndarray,
    max_move_sec: float,
) -> dict[tuple[int, str], tuple[float, str]]:
    """Use one raw-EIC valley for touching/nearby adjacent human peak boxes."""
    overrides: dict[tuple[int, str], tuple[float, str]] = {}
    ordered = sorted(peak_infos, key=lambda value: float(rt[value["apex_idx"]]))
    max_move_min = float(max_move_sec) / 60.0
    clean_eic = np.asarray(eic, dtype=np.float64)
    clean_eic = np.where(np.isfinite(clean_eic), clean_eic, 0.0)
    for left, right in zip(ordered, ordered[1:]):
        left_apex = int(left["apex_idx"])
        right_apex = int(right["apex_idx"])
        if left_apex >= right_apex - 1:
            continue
        between = np.arange(left_apex + 1, right_apex, dtype=int)
        if between.size == 0:
            continue
        valley_idx = int(between[int(np.argmin(clean_eic[between]))])
        valley_height = float(clean_eic[valley_idx])
        left_height = float(clean_eic[left_apex])
        right_height = float(clean_eic[right_apex])
        # A shared boundary must be a real valley for both human boxes.  If
        # the "valley" is already as high as either smaller apex, extending
        # that box to the point would absorb the neighbouring rising edge and
        # can move its maximum onto the boundary.
        if valley_height >= min(left_height, right_height):
            continue
        valley_rt = float(rt[valley_idx])
        left_move = abs(valley_rt - float(left["old_right_rt"]))
        right_move = abs(valley_rt - float(right["old_left_rt"]))
        # Only join boundaries that were already drawn near the same inter-peak
        # valley.  This prevents unrelated distant boxes from being bridged.
        if left_move > max_move_min + 1e-12 or right_move > max_move_min + 1e-12:
            continue
        reason = "相邻多峰共用两峰之间的原始EIC谷底"
        overrides[(int(left["peak_index"]), "right")] = (valley_rt, reason)
        overrides[(int(right["peak_index"]), "left")] = (valley_rt, reason)
    return overrides


def _format_box(box: tuple[float, float, float, float]) -> str:
    return json.dumps([round(float(v), 3) for v in box], ensure_ascii=False)


def _sample_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    work = df[df["source_file"].isin(SELECTED_SOURCE_FILES)].copy()
    true_n = pd.to_numeric(work["n_true_peak_boxes"], errors="coerce").fillna(0).astype(int)
    out_n = pd.to_numeric(work["n_out_fig_boxes"], errors="coerce").fillna(0).astype(int)
    boxes = true_n + out_n
    is_frag = work["feature_row_source"].eq("frag1_pos20_1_batch")

    strata = [
        (work[out_n > 0], 20, seed + 1),
        (work[is_frag & (out_n == 0) & (boxes > 1)], 10, seed + 2),
        (work[is_frag & (out_n == 0) & (boxes == 1)], 10, seed + 3),
        (work[is_frag & (boxes == 0)], 10, seed + 4),
        (work[(~is_frag) & (out_n == 0) & (boxes > 1)], 30, seed + 5),
        (work[(~is_frag) & (out_n == 0) & (boxes == 1)], 50, seed + 6),
        (work[(~is_frag) & (boxes == 0)], 20, seed + 7),
    ]
    sampled = []
    for pool, count, state in strata:
        if len(pool) < count:
            raise RuntimeError(f"试运行分层样本不足：需要{count}，仅有{len(pool)}")
        sampled.append(pool.sample(n=count, random_state=state, replace=False))
    result = pd.concat(sampled, ignore_index=False)
    if result.index.duplicated().any():
        raise RuntimeError("分层抽样出现重复行")
    return result.sample(frac=1.0, random_state=seed + 99).reset_index(drop=True)


def _make_check_image(
    reference_rgb: np.ndarray,
    output_path: Path,
    row: pd.Series,
    rt: np.ndarray,
    eic: np.ndarray,
    axis: AxisSnapshot,
    original_boxes: list[tuple[float, float, float, float]],
    corrected_boxes: list[tuple[float, float, float, float]],
    peaks: list[dict[str, Any]],
    tolerance: float,
    tolerance_source: str,
    reference_source: str,
    mapping_source: str,
    mapping_center_rt: float,
    mapping_score: float,
    seed_label: int,
    seed_relationship: str,
    seed_decision_reason: str,
    seed_diagnostics: dict[str, Any],
) -> None:
    ui_font = font_manager.FontProperties(family=FONT_FAMILIES)
    legend_font = ui_font.copy()
    legend_font.set_size(7)
    old_rgb = np.asarray(reference_rgb, dtype=np.uint8)
    ih, iw = old_rgb.shape[:2]
    fig = plt.figure(figsize=(16, 9.5), dpi=140, constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.45, 8.55],
        width_ratios=[1.0, 1.0],
        wspace=0.12,
        hspace=0.04,
    )
    header_ax = fig.add_subplot(outer[0, :])
    header_ax.set_axis_off()
    left_ax = fig.add_subplot(outer[1, 0])
    right_grid = outer[1, 1].subgridspec(
        2,
        2,
        height_ratios=[11.0, max(1.5, 0.75 * max(1, len(peaks)))],
        width_ratios=[5.4, 1.3],
        hspace=0.08,
        wspace=0.04,
    )
    right_ax = fig.add_subplot(right_grid[0, 0])
    right_legend_ax = fig.add_subplot(right_grid[0, 1])
    right_legend_ax.set_axis_off()
    interval_ax = fig.add_subplot(right_grid[1, 0], sharex=right_ax)
    delta_ax = fig.add_subplot(right_grid[1, 1], sharey=interval_ax)
    delta_ax.set_axis_off()
    left_ax.imshow(old_rgb)
    left_ax.set_xlim(0, iw)
    left_ax.set_ylim(ih, 0)
    left_ax.set_title(
        "LabelMe原始标注图：红色粗虚线=原框；绿色细实线=修正框",
        fontproperties=ui_font,
    )

    left_ax.add_patch(
        Rectangle(
            (axis.plot_left, axis.plot_top),
            axis.plot_right - axis.plot_left,
            axis.plot_bottom - axis.plot_top,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            label="实际绘图区",
        )
    )
    rt_mask = (rt >= mapping_center_rt - 1.0) & (rt <= mapping_center_rt + 1.0)
    rt_plot = rt[rt_mask]
    eic_plot = eic[rt_mask]
    x_pixels = [_rt_to_pixel_x(axis.ax, value) for value in rt_plot]
    y_pixels = [_intensity_to_pixel_y(axis.ax, value, ih) for value in eic_plot]
    left_ax.plot(x_pixels, y_pixels, color="navy", linewidth=0.8, alpha=0.75, label="mzML原始EIC")
    left_ax.scatter(x_pixels, y_pixels, s=5, color="navy", alpha=0.55)

    for i, box in enumerate(original_boxes):
        x1, y1, x2, y2 = box
        left_ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="#d62728", linestyle=(0, (4, 3)),
                linewidth=3.0, zorder=6,
                label="原始人工框" if i == 0 else None,
            )
        )
    for i, box in enumerate(corrected_boxes):
        x1, y1, x2, y2 = box
        left_ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="#16883e", linewidth=1.15, zorder=7,
                label="修正后框" if i == 0 else None,
            )
        )
    seed_rt = float(row["RT"])
    seed_rtmin = float(row["RTmin"])
    seed_rtmax = float(row["RTmax"])
    for value, color, label in [
        (seed_rt, "magenta", "Seed RT"),
        (seed_rtmin, "orange", "Seed RTmin"),
        (seed_rtmax, "darkorange", "Seed RTmax"),
    ]:
        left_ax.axvline(_rt_to_pixel_x(axis.ax, value), color=color, linewidth=1.0, label=label)
    for i, peak in enumerate(peaks):
        apex_x = _rt_to_pixel_x(axis.ax, float(peak["峰顶RT"]))
        apex_y = _intensity_to_pixel_y(axis.ax, float(peak["峰高"]), ih)
        left_ax.scatter([apex_x], [apex_y], marker="*", s=80, color="gold", edgecolor="black", label="峰顶" if i == 0 else None)
    left_ax.legend(loc="upper left", prop=legend_font)

    right_ax.plot(rt_plot, eic_plot, color="royalblue", linewidth=1.0, label="EIC连线")
    right_ax.scatter(rt_plot, eic_plot, s=10, color="royalblue", label="EIC散点")
    right_ax.axvline(mapping_center_rt - 1.0, color="cyan", linewidth=1.5, label="绘图区边界")
    right_ax.axvline(mapping_center_rt + 1.0, color="cyan", linewidth=1.5)
    right_ax.axvline(seed_rt, color="magenta", linewidth=1.0, label="Seed RT")
    right_ax.axvline(seed_rtmin, color="orange", linewidth=1.0, label="Seed RTmin/max")
    right_ax.axvline(seed_rtmax, color="darkorange", linewidth=1.0)
    interval_positions: list[float] = []
    interval_labels: list[str] = []
    for i, peak in enumerate(peaks):
        old_left = float(peak["原始左边界RT"])
        old_right = float(peak["原始右边界RT"])
        new_left = float(peak["修正后左边界RT"])
        new_right = float(peak["修正后右边界RT"])
        right_ax.axvline(old_left, color="#d62728", linestyle=(0, (4, 3)), linewidth=2.5, zorder=4, label="原始左右边界" if i == 0 else None)
        right_ax.axvline(old_right, color="#d62728", linestyle=(0, (4, 3)), linewidth=2.5, zorder=4)
        right_ax.axvline(new_left, color="#16883e", linewidth=1.15, zorder=5, label="修正左右边界" if i == 0 else None)
        right_ax.axvline(new_right, color="#16883e", linewidth=1.15, zorder=5)
        right_ax.scatter([float(peak["峰顶RT"])], [float(peak["峰高"])], marker="*", s=90, color="gold", edgecolor="black", label="峰顶" if i == 0 else None)

        corrected_y = float(2 * i)
        original_y = corrected_y + 1.0
        interval_ax.hlines(
            original_y, old_left, old_right, color="#d62728",
            linewidth=3.0, linestyles=(0, (4, 3)), zorder=2,
        )
        interval_ax.vlines(
            [old_left, old_right], original_y - 0.18, original_y + 0.18,
            color="#d62728", linewidth=1.5, linestyles="--",
        )
        interval_ax.hlines(
            corrected_y, new_left, new_right, color="#16883e",
            linewidth=1.8, zorder=3,
        )
        interval_ax.vlines(
            [new_left, new_right], corrected_y - 0.18, corrected_y + 0.18,
            color="#16883e", linewidth=1.2,
        )
        interval_positions.extend([corrected_y, original_y])
        interval_labels.extend([f"P{i + 1} Corrected", f"P{i + 1} Original"])
        delta_ax.text(
            0.02,
            corrected_y,
            f"ΔL={(new_left - old_left) * 60.0:+.2f}s\n"
            f"ΔR={(new_right - old_right) * 60.0:+.2f}s",
            transform=delta_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            fontfamily="Arial",
            zorder=5,
        )
    right_ax.set_xlim(mapping_center_rt - 1.0, mapping_center_rt + 1.0)
    right_ax.set_ylabel("Intensity")
    right_ax.set_title("mzML重提取原始EIC（边界修正依据）", fontproperties=ui_font)
    legend_handles, legend_labels = right_ax.get_legend_handles_labels()
    right_legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        prop=legend_font,
        borderaxespad=0.0,
    )
    right_ax.grid(alpha=0.2)
    right_ax.tick_params(axis="x", labelbottom=False)
    for tick in [*right_ax.get_xticklabels(), *right_ax.get_yticklabels()]:
        tick.set_fontfamily("Arial")

    if interval_positions:
        interval_ax.set_yticks(interval_positions, labels=interval_labels)
        interval_ax.set_ylim(-0.6, max(interval_positions) + 0.6)
        delta_ax.set_ylim(interval_ax.get_ylim())
    else:
        interval_ax.set_yticks([])
        interval_ax.set_ylim(-0.5, 0.5)
        delta_ax.set_ylim(interval_ax.get_ylim())
        interval_ax.text(
            0.5, 0.5, "无人工真峰框", transform=interval_ax.transAxes,
            ha="center", va="center", fontproperties=ui_font, fontsize=8,
        )
    for tick in interval_ax.get_yticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontsize(7)
    for tick in interval_ax.get_xticklabels():
        tick.set_fontfamily("Arial")
    interval_ax.set_xlabel("RT (min)", fontfamily="Arial")
    interval_ax.set_title(
        "原始/修正区间分轨（横坐标为真实RT）",
        fontproperties=ui_font,
        fontsize=8,
    )
    delta_ax.set_title("边界变化", fontproperties=ui_font, fontsize=8)
    interval_ax.grid(axis="x", alpha=0.2)
    interval_ax.spines["top"].set_visible(False)
    interval_ax.spines["right"].set_visible(False)

    mapping = (
        f"RT→x: x={axis.plot_left:.2f}+(RT-({mapping_center_rt:.6f}-1))*"
        f"({axis.plot_right:.2f}-{axis.plot_left:.2f})/2; "
        f"plot=[{axis.plot_left:.2f},{axis.plot_top:.2f},{axis.plot_right:.2f},{axis.plot_bottom:.2f}]"
    )
    header_ax.text(
        0.5,
        0.98,
        f"{row['Feature_ID']} | {tolerance:g} ppm | 原始框=红色虚线，修正框=绿色实线\n"
        f"{mapping}\n{reference_source}；{mapping_source}；映射拟合={mapping_score:.3f}\n{tolerance_source}"
        f"\nSeed判定={seed_label}；关系={seed_relationship}；"
        f"原始Seed峰宽={float(seed_diagnostics['width_sec']):.2f}s，"
        f"峰高={float(seed_diagnostics['height']):.3g}，面积={float(seed_diagnostics['area']):.3g}\n"
        f"{seed_decision_reason}",
        ha="center",
        va="top",
        fontsize=9,
        fontproperties=ui_font,
        linespacing=1.22,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_csv = Path(args.feature_csv).resolve()
    images_root = Path(args.images_root).resolve()
    output_root = Path(args.output_root).resolve()
    is_full = str(args.mode) == "full"
    write_check_images = (
        bool(args.write_check_images)
        if args.write_check_images is not None
        else not is_full
    )
    copy_corrected_images = (
        bool(args.copy_corrected_images)
        if args.copy_corrected_images is not None
        else not is_full
    )
    embed_missing_image_data = (
        bool(args.embed_missing_image_data)
        if args.embed_missing_image_data is not None
        else is_full
    )
    mapping_mode = (
        "analytic" if str(args.mapping_mode) == "auto" and is_full
        else (
            "exhaustive"
            if str(args.mapping_mode) == "auto"
            else str(args.mapping_mode)
        )
    )
    backup_root = output_root / "原始标注备份"
    corrected_root = output_root / ("修正后标注" if is_full else "修正后标注_试运行")
    check_root = output_root / ("边界检查图" if is_full else "边界检查图_试运行")
    seed_csv = output_root / ("原始种子属性表.csv" if is_full else "原始种子属性表_试运行.csv")
    peak_csv = output_root / ("真峰实例属性表.csv" if is_full else "真峰实例属性表_试运行.csv")
    qa_json = output_root / ("全量配置与自检.json" if is_full else "试运行配置与自检.json")

    if output_root.exists() and any(output_root.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(f"输出目录已存在且非空，拒绝覆盖：{output_root}")
    directories = [output_root, backup_root, corrected_root]
    if write_check_images:
        directories.append(check_root)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    required = {
        "Feature_ID", "source_file", "source_path", "mz", "RT", "RTmin", "RTmax",
        "feature_row_source", "image_path", "json_path", "n_true_peak_boxes", "n_out_fig_boxes",
        *PEAK_ATTRIBUTE_COLUMNS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"原始特征表缺少列：{missing}")
    df = df.reset_index(drop=True)
    df["_global_row_no"] = np.arange(1, len(df) + 1, dtype=np.int64)
    selected_sources = [
        source.strip()
        for source in str(args.only_source_files).split(",")
        if source.strip()
    ]
    if is_full:
        sample = df.copy()
        if selected_sources:
            sample = sample[sample["source_file"].isin(selected_sources)].copy()
            missing_sources = sorted(set(selected_sources) - set(sample["source_file"]))
            if missing_sources:
                raise ValueError(f"指定来源不在特征表中：{missing_sources}")
        sample = sample.sort_values("_global_row_no").reset_index(drop=True)
        if int(args.limit) > 0:
            sample = sample.head(int(args.limit)).copy()
        if sample.empty:
            raise RuntimeError("全量处理范围为空")
    else:
        if selected_sources:
            raise ValueError("试运行模式不支持--only-source-files")
        sample = _sample_rows(df, seed=int(args.random_seed))
        if not 100 <= len(sample) <= 200:
            raise RuntimeError(f"试运行图片数越界：{len(sample)}")

    tolerance_policy = str(args.tolerance_policy)
    tolerance_map: dict[str, float] = dict(SOURCE_TOLERANCE_PPM)
    tolerance_map_source = "脚本内试运行来源映射"
    if tolerance_policy == "source-map":
        if str(args.source_tolerance_map).strip():
            tolerance_path = Path(args.source_tolerance_map).resolve()
            raw_tolerance_config = json.loads(
                tolerance_path.read_text(encoding="utf-8")
            )
            raw_mapping = raw_tolerance_config.get(
                "source_tolerance_ppm", raw_tolerance_config
            )
            if not isinstance(raw_mapping, dict):
                raise ValueError("来源容差配置必须是JSON对象")
            tolerance_map = {
                str(source): float(value)
                for source, value in raw_mapping.items()
            }
            tolerance_map_source = str(tolerance_path)
        missing_tolerance_sources = sorted(
            set(sample["source_file"].astype(str)) - set(tolerance_map)
        )
        if missing_tolerance_sources:
            raise ValueError(
                f"来源容差配置缺少{len(missing_tolerance_sources)}个mzML："
                f"{missing_tolerance_sources}"
            )
    else:
        tolerance_map = {
            str(source): float(args.tolerance_ppm)
            for source in sample["source_file"].astype(str).unique()
        }
        tolerance_map_source = f"全量统一配置{float(args.tolerance_ppm):g}ppm"

    mapping_center_cache: dict[str, dict[str, Any]] = {}
    if str(args.mapping_center_cache).strip():
        mapping_cache_path = Path(args.mapping_center_cache).resolve()
        mapping_cache_df = pd.read_csv(
            mapping_cache_path, dtype=str, keep_default_na=False
        )
        required_cache_columns = {"原始特征编号", "绘图中心RT"}
        missing_cache_columns = sorted(
            required_cache_columns - set(mapping_cache_df.columns)
        )
        if missing_cache_columns:
            raise ValueError(
                f"绘图中心缓存缺少列：{missing_cache_columns}"
            )
        for _, cache_row in mapping_cache_df.iterrows():
            cached_score = _safe_float(cache_row.get("映射拟合分数", np.nan))
            cached_bounds: tuple[float, float, float, float] | None = None
            if str(cache_row.get("绘图区坐标", "")).strip():
                values = json.loads(str(cache_row["绘图区坐标"]))
                if isinstance(values, list) and len(values) == 4:
                    cached_bounds = tuple(float(value) for value in values)
            mapping_center_cache[str(cache_row["原始特征编号"])] = {
                "center_rt": float(cache_row["绘图中心RT"]),
                "score": (
                    cached_score if np.isfinite(cached_score) else None
                ),
                "plot_bounds": cached_bounds,
                "ppm": _safe_float(
                    cache_row.get("EIC提取容差_ppm", np.nan)
                ),
            }

    original_paths: list[Path] = [input_csv]
    resolved_mzml: dict[str, Path] = {}
    sample_objects: list[dict[str, Any]] = []
    for image_no, (_, row) in enumerate(sample.iterrows(), start=1):
        json_path = images_root / row["json_path"]
        image_path = images_root / row["image_path"]
        if not json_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"图片或JSON缺失：{row['Feature_ID']}")
        source_file = str(row["source_file"])
        mzml = resolved_mzml.get(source_file)
        if mzml is None:
            mzml = _find_mzml(row["source_path"], source_file)
        if mzml is None:
            raise FileNotFoundError(f"mzML无法对应：{row['Feature_ID']} / {row['source_file']}")
        resolved_mzml[source_file] = mzml
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        rel_json = Path(row["json_path"])
        rel_image = Path(row["image_path"])
        backup_path = backup_root / rel_json
        corrected_json = corrected_root / rel_json
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, backup_path)
        corrected_image: Path | None = None
        reference_source = ""
        pixels_match_external = False
        if copy_corrected_images:
            corrected_image = corrected_root / rel_image
            corrected_image.parent.mkdir(parents=True, exist_ok=True)
            reference_rgb, reference_source, pixels_match_external = (
                _load_labelme_reference_image(obj, image_path)
            )
            if obj.get("imageData"):
                # The corrected sidecar must depict the same pixels on which
                # the LabelMe coordinates were drawn.
                Image.fromarray(reference_rgb).save(corrected_image, format="PNG")
            else:
                shutil.copy2(image_path, corrected_image)
        original_paths.extend([json_path, image_path])
        sample_objects.append(
            {
                "图片编号": (
                    f"IMG-{int(row['_global_row_no']):05d}"
                    if is_full
                    else f"IMG-{image_no:04d}"
                ),
                "row": row,
                "original_json": json_path,
                "original_image": image_path,
                "backup_json": backup_path,
                "corrected_json": corrected_json,
                "corrected_image": corrected_image,
                "obj": obj if not is_full else None,
                "mzml": mzml,
                "reference_source": reference_source,
                "pixels_match_external": pixels_match_external,
            }
        )
    original_paths.extend(sorted(set(resolved_mzml.values())))
    original_hash_before = {str(path): _sha256(path) for path in sorted(set(original_paths))}

    seed_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    stats = {
        "实际处理图片数": len(sample_objects),
        "True_Peak数量": 0,
        "OUT_FIG数量": 0,
        "10ppm数量": 0,
        "15ppm数量": 0,
        "其他ppm数量": 0,
        "无法回溯数量": 0,
        "X轴边界发生移动的真峰数量": 0,
        "OUT_FIG贴边数量": 0,
        "Seed单峰对应数量": 0,
        "Seed分裂多峰数量": 0,
        "Seed假峰数量": 0,
        "Seed肩峰复合框数量": 0,
        "图内额外真峰数量": 0,
        "需要人工复核的图片数量": 0,
        "需要人工复核的真峰数量": 0,
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in sample_objects:
        grouped.setdefault(str(item["mzml"]), []).append(item)

    mapping_cache_hits = 0
    for group_no, (mzml_text, items) in enumerate(grouped.items(), start=1):
        mzml_path = Path(mzml_text)
        print(f"[LOAD {group_no}/{len(grouped)}] {mzml_path.name} | samples={len(items)}", flush=True)
        spectra = load_ms1_spectra(mzml_path)
        if not spectra:
            raise RuntimeError(f"mzML中无MS1谱图：{mzml_path}")
        for item_no, item in enumerate(items, start=1):
            row: pd.Series = item["row"]
            source_obj = item.get("obj")
            if source_obj is None:
                source_obj = json.loads(
                    item["original_json"].read_text(encoding="utf-8")
                )
            obj: dict[str, Any] = copy.deepcopy(source_obj)
            image_no = item["图片编号"]
            feature_id = str(row["Feature_ID"])
            mz = float(row["mz"])
            seed_rt = float(row["RT"])
            reference_rgb, reference_source, pixels_match_external = _load_labelme_reference_image(
                obj, item["original_image"]
            )
            ih, iw = reference_rgb.shape[:2]
            chosen_ppm = float(tolerance_map[str(row["source_file"])])
            tolerance_source = (
                "经原图回溯验证的来源级固定映射"
                if tolerance_policy == "source-map"
                else tolerance_map_source
            )
            rt, eic, mass_track = _extract_trace(
                spectra, mz, chosen_ppm, unit="ppm", method="nearest"
            )
            cached_mapping = mapping_center_cache.get(feature_id)
            if cached_mapping is not None:
                cached_ppm = float(cached_mapping.get("ppm", np.nan))
                if np.isfinite(cached_ppm) and not np.isclose(
                    cached_ppm, chosen_ppm
                ):
                    # A plot center recovered from a different EIC trace is
                    # not safe to reuse.  Re-run source-image recovery when
                    # the audited ppm policy corrects an earlier pilot value.
                    cached_mapping = None
            if cached_mapping is not None:
                mapping_cache_hits += 1
                mapping_center_rt = float(cached_mapping["center_rt"])
                cached_bounds = cached_mapping.get("plot_bounds")
                if cached_bounds is None:
                    raise ValueError(
                        f"绘图中心缓存缺少绘图区坐标：{feature_id}"
                    )
                axis, _ = _build_analytic_axis_snapshot(
                    rt,
                    eic,
                    mapping_center_rt,
                    reference_rgb,
                    plot_bounds=cached_bounds,
                )
                cached_score = cached_mapping.get("score")
                mapping_score = (
                    float(cached_score)
                    if cached_score is not None
                    else _curve_mask_fit_score(
                        _curve_mask(reference_rgb),
                        _build_analytic_axis_snapshot(
                            rt,
                            eic,
                            mapping_center_rt,
                            reference_rgb,
                            plot_bounds=cached_bounds,
                        )[1],
                    )
                )
                mapping_source = "沿用已人工确认试运行的绘图中心RT"
            else:
                recovery = {
                    "analytic": _recover_original_axis_center_analytic,
                    "fast": _recover_original_axis_center_fast,
                    "exhaustive": _recover_original_axis_center,
                }[mapping_mode]
                (
                    axis,
                    mapping_center_rt,
                    mapping_source,
                    mapping_score,
                ) = recovery(
                    rt, eic, seed_rt, reference_rgb, iw, ih, int(args.image_dpi)
                )
            backtrace_ok = bool(mapping_score >= 0.25)
            if chosen_ppm == 10.0:
                stats["10ppm数量"] += 1
            elif chosen_ppm == 15.0:
                stats["15ppm数量"] += 1
            else:
                stats["其他ppm数量"] += 1
            if not backtrace_ok:
                stats["无法回溯数量"] += 1

            valid = _valid_shapes(obj)
            original_boxes: list[tuple[float, float, float, float]] = []
            corrected_boxes: list[tuple[float, float, float, float]] = []
            image_peak_rows: list[dict[str, Any]] = []
            image_reasons: list[str] = []
            if not backtrace_ok:
                image_reasons.append("LabelMe原始标注图EIC回溯拟合分数偏低")

            peak_infos: list[dict[str, Any]] = []
            for peak_index, (shape_index, shape) in enumerate(valid, start=1):
                label = str(shape.get("label", "")).strip()
                if label == "True_Peak":
                    stats["True_Peak数量"] += 1
                else:
                    stats["OUT_FIG数量"] += 1
                original_box = _normalize_box(shape)
                original_boxes.append(original_box)
                x_left, y_top_old, x_right, y_bottom_old = original_box
                old_left_rt = max(mapping_center_rt - 1.0, _pixel_x_to_rt(axis.ax, x_left, ih))
                old_right_rt = min(mapping_center_rt + 1.0, _pixel_x_to_rt(axis.ax, x_right, ih))
                if old_left_rt > old_right_rt:
                    old_left_rt, old_right_rt = old_right_rt, old_left_rt
                interval_mask = (rt >= old_left_rt) & (rt <= old_right_rt)
                if np.any(interval_mask):
                    interval_indices = np.flatnonzero(interval_mask)
                    apex_idx = int(interval_indices[int(np.argmax(eic[interval_mask]))])
                else:
                    apex_idx = _nearest_idx(rt, (old_left_rt + old_right_rt) / 2.0)
                peak_infos.append(
                    {
                        "peak_index": peak_index,
                        "shape_index": shape_index,
                        "label": label,
                        "original_box": original_box,
                        "old_left_rt": old_left_rt,
                        "old_right_rt": old_right_rt,
                        "apex_idx": apex_idx,
                    }
                )

            shared_overrides = _shared_valley_overrides(
                peak_infos, rt, eic, float(args.max_boundary_move_sec)
            )

            for info in peak_infos:
                peak_index = int(info["peak_index"])
                shape_index = int(info["shape_index"])
                label = str(info["label"])
                original_box = info["original_box"]
                x_left, y_top_old, x_right, y_bottom_old = original_box
                old_left_rt = float(info["old_left_rt"])
                old_right_rt = float(info["old_right_rt"])
                apex_idx = int(info["apex_idx"])

                out_side = "不适用"
                left_moved = right_moved = False
                left_reason = right_reason = ""
                new_left_rt = old_left_rt
                new_right_rt = old_right_rt
                left_override = shared_overrides.get((peak_index, "left"))
                right_override = shared_overrides.get((peak_index, "right"))
                if label == "OUT_FIG":
                    left_distance = abs(x_left - axis.plot_left)
                    right_distance = abs(axis.plot_right - x_right)
                    out_side = "左侧" if left_distance <= right_distance else "右侧"
                    if out_side == "左侧":
                        new_left_rt = mapping_center_rt - 1.0
                        left_moved = abs(new_left_rt - old_left_rt) > 1e-12
                        left_reason = "OUT_FIG贴到绘图区左边界"
                        if right_override is not None:
                            new_right_rt, right_reason = right_override
                            right_moved = abs(new_right_rt - old_right_rt) > 1e-12
                        else:
                            new_right_rt, right_moved, right_reason = _refine_shape_boundary(
                                rt, eic, old_right_rt, apex_idx, "right", float(args.max_boundary_move_sec)
                            )
                    else:
                        new_right_rt = mapping_center_rt + 1.0
                        right_moved = abs(new_right_rt - old_right_rt) > 1e-12
                        right_reason = "OUT_FIG贴到绘图区右边界"
                        if left_override is not None:
                            new_left_rt, left_reason = left_override
                            left_moved = abs(new_left_rt - old_left_rt) > 1e-12
                        else:
                            new_left_rt, left_moved, left_reason = _refine_shape_boundary(
                                rt, eic, old_left_rt, apex_idx, "left", float(args.max_boundary_move_sec)
                            )
                    stats["OUT_FIG贴边数量"] += 1
                else:
                    if left_override is not None:
                        new_left_rt, left_reason = left_override
                        left_moved = abs(new_left_rt - old_left_rt) > 1e-12
                    else:
                        new_left_rt, left_moved, left_reason = _refine_shape_boundary(
                            rt, eic, old_left_rt, apex_idx, "left", float(args.max_boundary_move_sec)
                        )
                    if right_override is not None:
                        new_right_rt, right_reason = right_override
                        right_moved = abs(new_right_rt - old_right_rt) > 1e-12
                    else:
                        new_right_rt, right_moved, right_reason = _refine_shape_boundary(
                            rt, eic, old_right_rt, apex_idx, "right", float(args.max_boundary_move_sec)
                        )

                peak_reasons: list[str] = []
                if new_left_rt >= float(rt[apex_idx]) or new_right_rt <= float(rt[apex_idx]):
                    new_left_rt, new_right_rt = old_left_rt, old_right_rt
                    left_moved = right_moved = False
                    peak_reasons.append("候选边界会越过峰顶，已回退原X边界")
                new_left_rt = max(mapping_center_rt - 1.0, min(new_left_rt, mapping_center_rt + 1.0))
                new_right_rt = max(mapping_center_rt - 1.0, min(new_right_rt, mapping_center_rt + 1.0))
                if new_left_rt > new_right_rt:
                    new_left_rt, new_right_rt = new_right_rt, new_left_rt

                final_mask = (rt >= new_left_rt) & (rt <= new_right_rt)
                if np.any(final_mask):
                    final_indices = np.flatnonzero(final_mask)
                    final_apex_idx = int(final_indices[int(np.argmax(eic[final_mask]))])
                else:
                    final_apex_idx = apex_idx
                    peak_reasons.append("修正区间内无扫描点")
                apex_rt = float(rt[final_apex_idx])
                if not (new_left_rt < apex_rt < new_right_rt):
                    # A newly included neighbouring rise can become the
                    # interval maximum exactly at a proposed boundary.  Keep
                    # the original human X-range in this ambiguous case
                    # instead of silently absorbing the adjacent signal.
                    new_left_rt, new_right_rt = old_left_rt, old_right_rt
                    left_moved = right_moved = False
                    left_reason = right_reason = (
                        "候选边界会使区间最大值落在边界，保持原人工X边界"
                    )
                    peak_reasons.append(
                        "候选边界会纳入相邻更高信号，已保守回退原人工X边界"
                    )
                    final_mask = (rt >= new_left_rt) & (rt <= new_right_rt)
                    if np.any(final_mask):
                        final_indices = np.flatnonzero(final_mask)
                        final_apex_idx = int(
                            final_indices[int(np.argmax(eic[final_mask]))]
                        )
                    else:
                        final_apex_idx = apex_idx
                    apex_rt = float(rt[final_apex_idx])

                x_move_left_sec = abs(new_left_rt - old_left_rt) * 60.0
                x_move_right_sec = abs(new_right_rt - old_right_rt) * 60.0
                ordinary_moves = []
                if not (label == "OUT_FIG" and out_side == "左侧"):
                    ordinary_moves.append(x_move_left_sec)
                if not (label == "OUT_FIG" and out_side == "右侧"):
                    ordinary_moves.append(x_move_right_sec)
                if any(value > float(args.max_boundary_move_sec) + 1e-6 for value in ordinary_moves):
                    peak_reasons.append("普通边界移动超过配置上限")

                peak_height = float(max(eic[final_apex_idx], 0.0))
                if peak_height <= 0:
                    peak_reasons.append("峰高为0")

                x_new_left = axis.plot_left if (label == "OUT_FIG" and out_side == "左侧") else _rt_to_pixel_x(axis.ax, new_left_rt)
                x_new_right = axis.plot_right if (label == "OUT_FIG" and out_side == "右侧") else _rt_to_pixel_x(axis.ax, new_right_rt)
                new_top = _intensity_to_pixel_y(axis.ax, peak_height, ih) - float(args.y_top_pad_px)
                new_bottom = _intensity_to_pixel_y(axis.ax, 0.0, ih) + float(args.y_bottom_pad_px)
                new_top = max(axis.plot_top, min(new_top, axis.plot_bottom))
                new_bottom = max(axis.plot_top, min(new_bottom, axis.plot_bottom))
                if new_top >= new_bottom:
                    peak_reasons.append("统一Y轴后高度无效")
                    new_top = axis.plot_top
                    new_bottom = axis.plot_bottom
                corrected_box = (
                    float(min(x_new_left, x_new_right)), float(new_top),
                    float(max(x_new_left, x_new_right)), float(new_bottom),
                )
                corrected_boxes.append(corrected_box)
                obj["shapes"][shape_index]["points"] = [
                    [corrected_box[0], corrected_box[1]],
                    [corrected_box[2], corrected_box[3]],
                ]
                if left_moved or right_moved:
                    stats["X轴边界发生移动的真峰数量"] += 1

                attrs = _compute_one_feature_attributes(
                    rt, eic, mass_track,
                    target_mz=mz,
                    target_rt_min=apex_rt,
                    target_rtmin=new_left_rt,
                    target_rtmax=new_right_rt,
                    rt_tol_sec=30.0,
                    include_literature_top=True,
                )
                if any(not np.isfinite(_safe_float(attrs.get(name))) for name in PEAK_ATTRIBUTE_COLUMNS):
                    peak_reasons.append("重新计算的16项属性含缺失值")
                if not backtrace_ok:
                    peak_reasons.extend(image_reasons)
                review = bool(peak_reasons)
                if review:
                    stats["需要人工复核的真峰数量"] += 1
                peak_id = (
                    f"{feature_id}__P{peak_index:02d}"
                    if is_full
                    else f"PTL-PILOT-{image_no.split('-')[-1]}-P{peak_index:02d}"
                )
                peak_row: dict[str, Any] = {
                    "真峰编号": peak_id,
                    "图片编号": image_no,
                    "图片路径": str(item["original_image"]),
                    "修正标注JSON路径": str(item["corrected_json"]),
                    "mzML文件": str(item["mzml"]),
                    "原始特征编号": feature_id,
                    "m/z": mz,
                    "图像宽度_px": int(iw),
                    "图像高度_px": int(ih),
                    "原始标签": label,
                    "检测训练标签": "真峰",
                    "检测类别编号": 1,
                    "是否OUT_FIG": "是" if label == "OUT_FIG" else "否",
                    "OUT_FIG靠边侧": out_side,
                    "原始框坐标": _format_box(original_box),
                    "原始框_xmin_px": float(original_box[0]),
                    "原始框_ymin_px": float(original_box[1]),
                    "原始框_xmax_px": float(original_box[2]),
                    "原始框_ymax_px": float(original_box[3]),
                    "修正后框坐标": _format_box(corrected_box),
                    "修正后框_xmin_px": float(corrected_box[0]),
                    "修正后框_ymin_px": float(corrected_box[1]),
                    "修正后框_xmax_px": float(corrected_box[2]),
                    "修正后框_ymax_px": float(corrected_box[3]),
                    "原始左边界RT": round(old_left_rt, 8),
                    "修正后左边界RT": round(new_left_rt, 8),
                    "峰顶RT": round(apex_rt, 8),
                    "原始右边界RT": round(old_right_rt, 8),
                    "修正后右边界RT": round(new_right_rt, 8),
                    "峰高": peak_height,
                    "EIC提取容差_ppm": chosen_ppm,
                    "对应Seed编号": "",
                    "是否主要对应峰": "否",
                    "是否需人工复核": "是" if review else "否",
                    "处理状态": "需人工复核" if review else "成功",
                    "异常原因": "；".join(dict.fromkeys(peak_reasons)),
                    "左边界修正规则": left_reason,
                    "右边界修正规则": right_reason,
                    "左边界移动秒": round(x_move_left_sec, 6),
                    "右边界移动秒": round(x_move_right_sec, 6),
                    "_peak_index": peak_index,
                }
                for name in PEAK_ATTRIBUTE_COLUMNS:
                    peak_row[name] = attrs.get(name, np.nan)
                image_peak_rows.append(peak_row)

            seed_rtmin = float(row["RTmin"])
            seed_rtmax = float(row["RTmax"])
            seed_diag = _seed_eic_diagnostics(
                rt, eic, seed_rt, seed_rtmin, seed_rtmax
            )
            seed_x1 = float(
                np.clip(
                    _rt_to_pixel_x(axis.ax, min(seed_rtmin, seed_rtmax)),
                    axis.plot_left,
                    axis.plot_right,
                )
            )
            seed_x2 = float(
                np.clip(
                    _rt_to_pixel_x(axis.ax, max(seed_rtmin, seed_rtmax)),
                    axis.plot_left,
                    axis.plot_right,
                )
            )
            if seed_x2 <= seed_x1:
                seed_center_x = float(
                    np.clip(
                        _rt_to_pixel_x(axis.ax, seed_rt),
                        axis.plot_left,
                        axis.plot_right,
                    )
                )
                seed_x1 = max(float(axis.plot_left), seed_center_x - 0.5)
                seed_x2 = min(float(axis.plot_right), seed_center_x + 0.5)
            seed_signal_height = max(float(seed_diag["height"]), 0.0)
            seed_y1 = float(
                np.clip(
                    _intensity_to_pixel_y(axis.ax, seed_signal_height, ih)
                    - float(args.y_top_pad_px),
                    axis.plot_top,
                    axis.plot_bottom,
                )
            )
            seed_y2 = float(
                np.clip(
                    _intensity_to_pixel_y(axis.ax, 0.0, ih)
                    + float(args.y_bottom_pad_px),
                    axis.plot_top,
                    axis.plot_bottom,
                )
            )
            if seed_y2 <= seed_y1:
                seed_y1 = max(float(axis.plot_top), seed_y2 - 1.0)
            original_seed_box = (seed_x1, seed_y1, seed_x2, seed_y2)
            info_by_index = {int(info["peak_index"]): info for info in peak_infos}
            peak_row_by_index = {
                int(peak["_peak_index"]): peak for peak in image_peak_rows
            }
            seed_match_tolerance_min = float(args.seed_match_tolerance_sec) / 60.0
            anchor_indices = {
                index
                for index, peak in peak_row_by_index.items()
                if float(peak["修正后左边界RT"]) - seed_match_tolerance_min
                <= seed_rt
                <= float(peak["修正后右边界RT"]) + seed_match_tolerance_min
            }
            overlap_indices = {
                index
                for index, peak in peak_row_by_index.items()
                if float(peak["修正后右边界RT"]) >= seed_rtmin
                and float(peak["修正后左边界RT"]) <= seed_rtmax
            }
            core_indices = {
                index
                for index in overlap_indices
                if seed_rtmin <= float(rt[info_by_index[index]["apex_idx"]]) <= seed_rtmax
            }
            shared_groups: dict[float, set[int]] = {}
            for (index, _side), (valley_rt, _reason) in shared_overrides.items():
                shared_groups.setdefault(round(float(valley_rt), 10), set()).add(int(index))
            shared_neighbors: dict[int, set[int]] = {
                index: set() for index in peak_row_by_index
            }
            for members in shared_groups.values():
                for index in members:
                    shared_neighbors.setdefault(index, set()).update(members - {index})

            def apex_outside_seed_seconds(index: int) -> float:
                apex_value_rt = float(rt[info_by_index[index]["apex_idx"]])
                if apex_value_rt < seed_rtmin:
                    return float((seed_rtmin - apex_value_rt) * 60.0)
                if apex_value_rt > seed_rtmax:
                    return float((apex_value_rt - seed_rtmax) * 60.0)
                return 0.0

            matched_indices: set[int] = set()
            seed_decision_reason = ""
            shoulder_relation = False
            shoulder_evidence = "不适用"
            seed_link_review = False
            normal_shift_relation = False
            if not image_peak_rows:
                seed_decision_reason = "图中无人工真峰框，Seed判为0"
            elif core_indices:
                matched_indices = set(core_indices)
                # Only a peak connected by the same inter-peak valley and
                # whose apex is at most 3 s beyond the original Seed interval
                # may extend a split Seed.  A box that merely grazes the Seed
                # interval remains an unlinked true peak in the same image.
                changed = True
                while changed:
                    changed = False
                    for index in overlap_indices - matched_indices:
                        if apex_outside_seed_seconds(index) > float(args.seed_match_tolerance_sec) + 1e-9:
                            continue
                        if shared_neighbors.get(index, set()) & matched_indices:
                            matched_indices.add(index)
                            changed = True
                seed_decision_reason = (
                    "至少一个人工峰顶位于原始Seed区间；仅将峰顶小幅越界且通过共享谷底相连的框计入Seed分裂"
                )
                if any(
                    apex_outside_seed_seconds(index) > 2.0
                    for index in matched_indices
                ):
                    seed_link_review = True
            elif not anchor_indices:
                seed_decision_reason = (
                    f"无人工峰顶落入原始Seed区间，且修正框未覆盖或接近Seed RT；"
                    "图中真峰作为额外真峰保留，Seed判为0"
                )
            else:
                shoulder_candidates = sorted(
                    anchor_indices & overlap_indices,
                    key=lambda index: abs(
                        float(rt[info_by_index[index]["apex_idx"]]) - seed_rt
                    ),
                )
                if not shoulder_candidates:
                    seed_decision_reason = (
                        "人工框仅在3秒容差内接近Seed RT但不与原始Seed区间重叠，Seed判为0"
                    )
                else:
                    closest_index = int(shoulder_candidates[0])
                    candidate_apex_height = float(
                        max(eic[info_by_index[closest_index]["apex_idx"]], 0.0)
                    )
                    seed_to_candidate_ratio = (
                        float(seed_diag["height"]) / candidate_apex_height
                        if candidate_apex_height > 0
                        and np.isfinite(float(seed_diag["height"]))
                        else float("nan")
                    )
                    outside_seconds = apex_outside_seed_seconds(closest_index)
                    if outside_seconds <= float(args.seed_match_tolerance_sec) and seed_to_candidate_ratio >= 0.50:
                        matched_indices = {closest_index}
                        normal_shift_relation = True
                        seed_decision_reason = (
                            "人工峰顶仅轻微越出原始Seed区间，且Seed区间峰高与人工峰同量级，按单峰对应"
                        )
                    else:
                        shoulder_evidence = _shoulder_evidence_status(
                            seed_diag, seed_to_candidate_ratio
                        )
                        if shoulder_evidence in {"pass", "uncertain"}:
                            matched_indices = {closest_index}
                            shoulder_relation = True
                            seed_link_review = shoulder_evidence == "uncertain"
                            seed_decision_reason = (
                                "人工框覆盖Seed自身信号，但框内全局峰顶属于相邻高峰；"
                                + (
                                    "肩峰证据通过，按肩峰-复合框判Seed为1"
                                    if shoulder_evidence == "pass"
                                    else "肩峰证据偏弱，暂判Seed为1并标记人工复核"
                                )
                            )
                        else:
                            seed_decision_reason = (
                                "人工框虽覆盖Seed RT，但Seed自身信号过低或截断，"
                                "图中高峰作为额外真峰保留，Seed判为0"
                            )

            matched_apex_height = max(
                (
                    float(max(eic[info_by_index[index]["apex_idx"]], 0.0))
                    for index in matched_indices
                ),
                default=0.0,
            )
            diagnostic_candidate_height = matched_apex_height
            if diagnostic_candidate_height <= 0 and anchor_indices:
                diagnostic_candidate_height = max(
                    (
                        float(max(eic[info_by_index[index]["apex_idx"]], 0.0))
                        for index in anchor_indices
                    ),
                    default=0.0,
                )
            seed_to_annotated_height_ratio = (
                float(seed_diag["height"]) / diagnostic_candidate_height
                if diagnostic_candidate_height > 0 and np.isfinite(float(seed_diag["height"]))
                else float("nan")
            )

            matched_peak_rows = [
                peak for peak in image_peak_rows
                if int(peak["_peak_index"]) in matched_indices
            ]
            for peak in matched_peak_rows:
                peak["对应Seed编号"] = feature_id
            primary: dict[str, Any] | None = None
            highest_peak: dict[str, Any] | None = None
            if matched_peak_rows:
                primary = min(
                    matched_peak_rows,
                    key=lambda value: abs(float(value["峰顶RT"]) - seed_rt),
                )
                primary["是否主要对应峰"] = "是"
                highest_peak = max(
                    matched_peak_rows,
                    key=lambda value: float(value["峰高"]),
                )

            match_count = len(matched_peak_rows)
            extra_peak_rows = [
                peak for peak in image_peak_rows
                if int(peak["_peak_index"]) not in matched_indices
            ]
            stats["图内额外真峰数量"] += len(extra_peak_rows)
            if match_count == 0:
                relationship = "假峰"
                stats["Seed假峰数量"] += 1
            elif shoulder_relation:
                relationship = "肩峰-复合框"
                stats["Seed肩峰复合框数量"] += 1
            elif match_count == 1:
                relationship = "单峰对应"
                stats["Seed单峰对应数量"] += 1
            else:
                relationship = "Seed分裂多峰"
                stats["Seed分裂多峰数量"] += 1

            highest_id = highest_peak["真峰编号"] if highest_peak is not None else ""
            for peak in image_peak_rows:
                peak_index_value = int(peak["_peak_index"])
                apex_value_rt = float(peak["峰顶RT"])
                apex_in_seed = bool(seed_rtmin <= apex_value_rt <= seed_rtmax)
                box_overlaps_seed = bool(
                    float(peak["修正后右边界RT"]) >= seed_rtmin
                    and float(peak["修正后左边界RT"]) <= seed_rtmax
                )
                outside_seconds = apex_outside_seed_seconds(peak_index_value)
                linked = peak_index_value in matched_indices
                if not linked:
                    association_type = "图内额外真峰"
                    association_reason = "未满足Seed核心峰、共享谷底分裂成员或肩峰复合框条件"
                elif shoulder_relation:
                    association_type = "肩峰复合框"
                    association_reason = "人工框覆盖Seed自身信号，肩峰/复合框证据成立"
                elif match_count > 1:
                    association_type = "Seed分裂成员"
                    association_reason = (
                        "峰顶位于原始Seed区间，或峰顶小幅越界且与Seed核心峰共享谷底"
                    )
                else:
                    association_type = "单峰对应"
                    association_reason = (
                        "峰顶位于原始Seed区间"
                        if apex_in_seed
                        else "峰顶仅轻微越界且Seed信号与人工峰同量级"
                    )
                peak["Seed关联类型"] = association_type
                peak["与原始Seed RT差_秒"] = round(abs(apex_value_rt - seed_rt) * 60.0, 6)
                peak["峰顶是否在原始Seed区间"] = "是" if apex_in_seed else "否"
                peak["峰顶越出原始Seed区间_秒"] = round(outside_seconds, 6)
                peak["与原始Seed区间是否重叠"] = "是" if box_overlaps_seed else "否"
                peak["是否匹配峰中最高峰"] = (
                    "不适用"
                    if not linked
                    else ("是" if str(peak["真峰编号"]) == str(highest_id) else "否")
                )
                peak["Seed关联依据"] = association_reason

            image_review = bool(
                seed_link_review
                or image_reasons
                or any(p["是否需人工复核"] == "是" for p in image_peak_rows)
            )
            if image_review:
                stats["需要人工复核的图片数量"] += 1
            seed_abnormal_reasons = list(image_reasons)
            seed_abnormal_reasons.extend(
                str(p["异常原因"])
                for p in image_peak_rows
                if str(p.get("异常原因", "")).strip()
            )
            if seed_link_review:
                seed_abnormal_reasons.append("Seed关联证据处于边缘情况，需人工复核")
            matched_ids = ";".join(p["真峰编号"] for p in matched_peak_rows)
            primary_id = next((p["真峰编号"] for p in matched_peak_rows if p["是否主要对应峰"] == "是"), "")
            extra_ids = ";".join(p["真峰编号"] for p in extra_peak_rows)

            if not obj.get("imageData") and embed_missing_image_data:
                obj["imageData"] = base64.b64encode(
                    item["original_image"].read_bytes()
                ).decode("ascii")
                obj["imagePath"] = item["original_image"].name
            elif (
                not obj.get("imageData")
                and not copy_corrected_images
            ):
                obj["imagePath"] = str(item["original_image"])

            item["corrected_json"].write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            item["corrected_boxes"] = list(corrected_boxes)
            for peak in image_peak_rows:
                peak.pop("_peak_index", None)
            peak_rows.extend(image_peak_rows)
            if primary is None:
                corrected_seed_rt = corrected_seed_left = corrected_seed_right = np.nan
                corrected_seed_union_left = corrected_seed_union_right = np.nan
                corrected_seed_boundary_source = "Seed=0，无修正后Seed边界"
            elif shoulder_relation:
                # A composite box's global apex/bounds describe the dominant
                # neighbour, not the shoulder Seed.  Preserve the Seed's own
                # original component geometry for Seed-level fusion.
                corrected_seed_rt = seed_rt
                corrected_seed_left = seed_rtmin
                corrected_seed_right = seed_rtmax
                corrected_seed_union_left = min(float(p["修正后左边界RT"]) for p in matched_peak_rows)
                corrected_seed_union_right = max(float(p["修正后右边界RT"]) for p in matched_peak_rows)
                corrected_seed_boundary_source = "肩峰/复合框：Seed自身边界沿用原Seed；检测总边界取匹配框并集"
            else:
                corrected_seed_rt = float(primary["峰顶RT"])
                corrected_seed_left = float(primary["修正后左边界RT"])
                corrected_seed_right = float(primary["修正后右边界RT"])
                corrected_seed_union_left = min(float(p["修正后左边界RT"]) for p in matched_peak_rows)
                corrected_seed_union_right = max(float(p["修正后右边界RT"]) for p in matched_peak_rows)
                corrected_seed_boundary_source = (
                    "主要对应峰按峰顶RT距原始Seed RT最近选择，其修正框作为Seed主边界；"
                    "总边界为全部匹配真峰并集；最高峰不自动替代主要峰"
                )
            seed_row: dict[str, Any] = {
                "图片编号": image_no,
                "图片路径": str(item["original_image"]),
                "原始标注JSON路径": str(item["original_json"]),
                "修正标注JSON路径": str(item["corrected_json"]),
                "mzML文件": str(item["mzml"]),
                "原始特征编号": feature_id,
                "m/z": row["mz"],
                "图像宽度_px": int(iw),
                "图像高度_px": int(ih),
                "原始Seed框坐标_px": _format_box(original_seed_box),
                "原始Seed框_xmin_px": float(original_seed_box[0]),
                "原始Seed框_ymin_px": float(original_seed_box[1]),
                "原始Seed框_xmax_px": float(original_seed_box[2]),
                "原始Seed框_ymax_px": float(original_seed_box[3]),
                "训练Seed框定义": "原始RTmin/RTmax与原始EIC信号生成，不使用人工真峰框",
                "原始种子RT": row["RT"],
                "原始左边界RT": row["RTmin"],
                "原始右边界RT": row["RTmax"],
                "修正后Seed RT": corrected_seed_rt,
                "修正后Seed左边界RT": corrected_seed_left,
                "修正后Seed右边界RT": corrected_seed_right,
                "修正后Seed总左边界RT": corrected_seed_union_left,
                "修正后Seed总右边界RT": corrected_seed_union_right,
                "修正后Seed边界来源": corrected_seed_boundary_source,
                "EIC提取容差_ppm": chosen_ppm,
                "容差来源": tolerance_source,
                "是否成功回溯": "是" if backtrace_ok else "否",
                "种子真假标签": 1 if match_count > 0 else 0,
                "匹配峰数量": match_count,
                "Seed匹配真峰数量": match_count,
                "种子关系": relationship,
                "Seed关系类型": relationship,
                "匹配真峰编号": matched_ids,
                "主要对应真峰编号": primary_id,
                "匹配峰中最高峰编号": highest_id,
                "主要对应峰选择依据": (
                    "峰顶RT距原始Seed RT最近；不默认选择最高峰"
                    if match_count > 0 else "Seed=0，无主要对应峰"
                ),
                "主要对应峰是否为最高峰": (
                    "不适用"
                    if match_count == 0
                    else ("是" if primary_id == highest_id else "否")
                ),
                "图内真峰总数": len(image_peak_rows),
                "图内额外真峰数量": len(extra_peak_rows),
                "图内额外真峰编号": extra_ids,
                "肩峰证据状态": {
                    "pass": "通过",
                    "uncertain": "需复核",
                    "fail": "未通过",
                }.get(shoulder_evidence, shoulder_evidence),
                "Seed关联是否需复核": "是" if seed_link_review else "否",
                "Seed判断依据": seed_decision_reason,
                "原始Seed峰宽_秒": round(float(seed_diag["width_sec"]), 6),
                "原始Seed峰高": float(seed_diag["height"]),
                "原始Seed峰高RT": round(float(seed_diag["height_rt"]), 8) if np.isfinite(float(seed_diag["height_rt"])) else np.nan,
                "原始Seed面积_强度乘秒": float(seed_diag["area"]),
                "原始Seed_RT处强度": float(seed_diag["seed_intensity"]),
                "Seed区间最大值是否卡边": "是" if seed_diag["max_at_edge"] else "否",
                "Seed峰高与对应人工峰峰高比": seed_to_annotated_height_ratio,
                "是否需人工复核": "是" if image_review else "否",
                "处理状态": "需人工复核" if image_review else "成功",
                "异常原因": "；".join(dict.fromkeys(seed_abnormal_reasons)),
                "绘图区坐标": _format_box((axis.plot_left, axis.plot_top, axis.plot_right, axis.plot_bottom)),
                "RT与像素映射": (
                    f"x={axis.plot_left:.6f}+(RT-({mapping_center_rt:.8f}-1))*"
                    f"({axis.plot_right:.6f}-{axis.plot_left:.6f})/2"
                ),
                "原始标注图来源": reference_source,
                "外部PNG与内嵌图是否一致": "是" if pixels_match_external else "否",
                "绘图中心RT": round(mapping_center_rt, 8),
                "绘图中心来源": mapping_source,
                "映射拟合分数": round(mapping_score, 6),
            }
            for name in PEAK_ATTRIBUTE_COLUMNS:
                seed_row[f"原始{name}"] = row[name]
            seed_rows.append(seed_row)

            if write_check_images:
                check_path = check_root / Path(row["image_path"]).with_suffix(".png")
                _make_check_image(
                    reference_rgb, check_path, row, rt, eic, axis,
                    original_boxes, corrected_boxes, image_peak_rows,
                    chosen_ppm, tolerance_source, reference_source, mapping_source,
                    mapping_center_rt, mapping_score,
                    1 if match_count > 0 else 0, relationship,
                    seed_decision_reason, seed_diag,
                )
            if axis.fig is not None:
                plt.close(axis.fig)
            if (
                item_no == 1
                or item_no == len(items)
                or item_no % max(1, int(args.progress_every)) == 0
            ):
                print(
                    f"  [{item_no}/{len(items)}] {feature_id} | "
                    f"true_boxes={len(image_peak_rows)} | matched={match_count} | "
                    f"ppm={chosen_ppm:g}",
                    flush=True,
                )

    seed_df = pd.DataFrame(seed_rows)
    peak_df = pd.DataFrame(peak_rows)
    seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    peak_df.to_csv(peak_csv, index=False, encoding="utf-8-sig")

    original_hash_after = {path: _sha256(Path(path)) for path in original_hash_before}
    unchanged = original_hash_before == original_hash_after
    backup_exact = all(
        _sha256(item["original_json"]) == _sha256(item["backup_json"])
        for item in sample_objects
    )
    corrected_valid = True
    corrected_png_aligned = True
    corrected_coordinates_match = True
    corrected_shape_count = 0
    original_shape_count = 0
    for item in sample_objects:
        original_obj = item.get("obj")
        if original_obj is None:
            original_obj = json.loads(
                item["original_json"].read_text(encoding="utf-8")
            )
        corrected_obj = json.loads(item["corrected_json"].read_text(encoding="utf-8"))
        original_valid = _valid_shapes(original_obj)
        corrected_valid_shapes = _valid_shapes(corrected_obj)
        original_shape_count += len(original_valid)
        corrected_shape_count += len(corrected_valid_shapes)
        original_labels = [str(shape.get("label", "")) for _, shape in original_valid]
        corrected_labels = [str(shape.get("label", "")) for _, shape in corrected_valid_shapes]
        if len(original_valid) != len(corrected_valid_shapes) or original_labels != corrected_labels:
            corrected_valid = False
        corrected_image = item.get("corrected_image")
        image_reference_ok = bool(corrected_obj.get("imageData"))
        if not image_reference_ok and corrected_image is not None:
            image_reference_ok = corrected_image.exists()
        if not image_reference_ok:
            image_path_value = Path(str(corrected_obj.get("imagePath", "")))
            if not image_path_value.is_absolute():
                image_path_value = item["corrected_json"].parent / image_path_value
            image_reference_ok = image_path_value.exists()
        if not image_reference_ok:
            corrected_valid = False
        if copy_corrected_images:
            if corrected_image is None or not corrected_image.exists():
                corrected_png_aligned = False
            else:
                corrected_rgb = np.asarray(
                    Image.open(corrected_image).convert("RGB")
                )
                reference_rgb, _, _ = _load_labelme_reference_image(
                    original_obj, item["original_image"]
                )
                if (
                    corrected_rgb.shape != reference_rgb.shape
                    or not np.array_equal(corrected_rgb, reference_rgb)
                ):
                    corrected_png_aligned = False
        corrected_boxes_from_json = [_normalize_box(shape) for _, shape in corrected_valid_shapes]
        expected_boxes = item.get("corrected_boxes", [])
        if len(corrected_boxes_from_json) != len(expected_boxes) or any(
            not np.allclose(got, expected, rtol=0.0, atol=1e-8)
            for got, expected in zip(corrected_boxes_from_json, expected_boxes)
        ):
            corrected_coordinates_match = False

    left_is_out_fig_edge = (
        (peak_df["原始标签"] == "OUT_FIG")
        & (peak_df["OUT_FIG靠边侧"] == "左侧")
    )
    right_is_out_fig_edge = (
        (peak_df["原始标签"] == "OUT_FIG")
        & (peak_df["OUT_FIG靠边侧"] == "右侧")
    )
    left_move_ok = left_is_out_fig_edge | (
        pd.to_numeric(peak_df["左边界移动秒"])
        <= float(args.max_boundary_move_sec) + 1e-6
    )
    right_move_ok = right_is_out_fig_edge | (
        pd.to_numeric(peak_df["右边界移动秒"])
        <= float(args.max_boundary_move_sec) + 1e-6
    )
    boundary_order_ok = (
        (pd.to_numeric(peak_df["修正后左边界RT"]) < pd.to_numeric(peak_df["峰顶RT"]))
        & (pd.to_numeric(peak_df["峰顶RT"]) < pd.to_numeric(peak_df["修正后右边界RT"]))
    )
    ordinary_move_valid = bool((left_move_ok & right_move_ok & boundary_order_ok).all())
    seed_relation_valid = True
    for _, seed_record in seed_df.iterrows():
        fid = str(seed_record["原始特征编号"])
        linked = peak_df[peak_df["对应Seed编号"].astype(str) == fid]
        expected = int(seed_record["匹配峰数量"])
        label = int(seed_record["种子真假标签"])
        if len(linked) != expected or label != int(expected > 0):
            seed_relation_valid = False
            break
    seed_highest_semantics_valid = bool(
        (
            (pd.to_numeric(seed_df["Seed匹配真峰数量"]) == 0)
            == (seed_df["主要对应峰是否为最高峰"].astype(str) == "不适用")
        ).all()
    )
    peak_highest_semantics_valid = bool(
        (
            peak_df["对应Seed编号"].fillna("").astype(str).eq("")
            == peak_df["是否匹配峰中最高峰"].astype(str).eq("不适用")
        ).all()
    )

    def target_seed(feature_id: str) -> pd.Series | None:
        records = seed_df[seed_df["原始特征编号"].astype(str) == feature_id]
        return records.iloc[0] if len(records) == 1 else None

    f50_seed = target_seed("060-0145-006_018__F50")
    f50_peaks = (
        peak_df[peak_df["图片编号"] == f50_seed["图片编号"]]
        if f50_seed is not None else peak_df.iloc[0:0]
    )
    f50_shape_ok = bool(
        f50_seed is not None
        and len(f50_peaks) == 1
        and abs(float(f50_peaks.iloc[0]["修正后右边界RT"]) - 9.523366666667) < 1e-6
        and "向外小幅扩展" in str(f50_peaks.iloc[0]["右边界修正规则"])
    )
    f11629_seed = target_seed("0001_MAY_RoCI-StEM_CP-287__F11629")
    f11629_peaks = (
        peak_df[peak_df["图片编号"] == f11629_seed["图片编号"]]
        if f11629_seed is not None else peak_df.iloc[0:0]
    )
    f11629_noise_guard_ok = bool(
        f11629_seed is not None
        and len(f11629_peaks) == 1
        and abs(
            float(f11629_peaks.iloc[0]["修正后右边界RT"])
            - float(f11629_peaks.iloc[0]["原始右边界RT"])
        ) < 1e-8
    )
    f1209_seed = target_seed("0001_MAY_RoCI-StEM_CP-287__F1209")
    f1209_peaks = (
        peak_df[peak_df["图片编号"] == f1209_seed["图片编号"]].sort_values("峰顶RT")
        if f1209_seed is not None else peak_df.iloc[0:0]
    )
    f1209_split_ok = bool(
        f1209_seed is not None
        and str(f1209_seed["Seed关系类型"]) == "Seed分裂多峰"
        and int(f1209_seed["Seed匹配真峰数量"]) == 2
        and len(f1209_peaks) == 2
        and abs(
            float(f1209_peaks.iloc[0]["修正后右边界RT"])
            - float(f1209_peaks.iloc[1]["修正后左边界RT"])
        ) < 1e-8
    )
    f745_seed = target_seed("0001_MAY_RoCI-StEM_CP-287__F745")
    f745_false_ok = bool(
        f745_seed is not None
        and int(f745_seed["种子真假标签"]) == 0
        and int(f745_seed["Seed匹配真峰数量"]) == 0
        and int(f745_seed["图内额外真峰数量"]) >= 1
    )
    f345_seed = target_seed("20180321_S00033936_P__F345")
    f345_extra_ok = bool(
        f345_seed is not None
        and str(f345_seed["Seed关系类型"]) == "单峰对应"
        and int(f345_seed["Seed匹配真峰数量"]) == 1
        and int(f345_seed["图内额外真峰数量"]) >= 1
    )
    f514_seed = target_seed("20180321_S00033936_P__F514")
    f514_extra_ok = bool(
        f514_seed is not None
        and str(f514_seed["Seed关系类型"]) == "单峰对应"
        and int(f514_seed["Seed匹配真峰数量"]) == 1
        and int(f514_seed["图内额外真峰数量"]) >= 1
    )
    logic_checks = _logic_self_checks()

    checks = {
        (
            "全量处理行数等于选定特征表行数"
            if is_full
            else "抽样图片数在100到200之间"
        ): (
            bool(len(seed_df) == len(sample))
            if is_full
            else bool(100 <= len(seed_df) <= 200)
        ),
        "原始文件SHA256未变化": unchanged,
        "原始JSON备份逐字节一致": backup_exact,
        "修正JSON结构和标签可供LabelMe加载": corrected_valid,
        (
            "全量模式未复制PNG"
            if not copy_corrected_images
            else "修正目录PNG与LabelMe原始标注图像素一致"
        ): (
            bool(not list(corrected_root.rglob("*.png")))
            if not copy_corrected_images
            else corrected_png_aligned
        ),
        "修正JSON坐标与真峰表一致": corrected_coordinates_match,
        "普通边界可收缩或外扩但均不超过3秒且不越峰顶": ordinary_move_valid,
        "Seed标签匹配数量与真峰反向关联一致": seed_relation_valid,
        "无匹配Seed的最高峰比较明确写为不适用": seed_highest_semantics_valid,
        "图内额外真峰的匹配最高峰比较明确写为不适用": peak_highest_semantics_valid,
        **logic_checks,
        "多框未丢失": bool(original_shape_count == corrected_shape_count == len(peak_df)),
        "种子表行数等于图片数": bool(len(seed_df) == len(sample_objects)),
        "真峰表行数等于有效框数": bool(len(peak_df) == original_shape_count),
        "Seed和真峰编号唯一": bool(
            seed_df["原始特征编号"].astype(str).is_unique
            and peak_df["真峰编号"].astype(str).is_unique
        ),
        "原始Seed训练框合法且位于图内": bool(
            (
                pd.to_numeric(seed_df["原始Seed框_xmin_px"]).ge(0)
                & pd.to_numeric(seed_df["原始Seed框_ymin_px"]).ge(0)
                & (
                    pd.to_numeric(seed_df["原始Seed框_xmax_px"])
                    > pd.to_numeric(seed_df["原始Seed框_xmin_px"])
                )
                & (
                    pd.to_numeric(seed_df["原始Seed框_ymax_px"])
                    > pd.to_numeric(seed_df["原始Seed框_ymin_px"])
                )
                & (
                    pd.to_numeric(seed_df["原始Seed框_xmax_px"])
                    <= pd.to_numeric(seed_df["图像宽度_px"])
                )
                & (
                    pd.to_numeric(seed_df["原始Seed框_ymax_px"])
                    <= pd.to_numeric(seed_df["图像高度_px"])
                )
            ).all()
        ),
        "两张CSV为UTF-8-SIG": bool(
            seed_csv.read_bytes().startswith(b"\xef\xbb\xbf")
            and peak_csv.read_bytes().startswith(b"\xef\xbb\xbf")
        ),
        (
            "检查图数量等于图片数"
            if write_check_images
            else "全量模式未输出检查图"
        ): (
            bool(len(list(check_root.rglob("*.png"))) == len(sample_objects))
            if write_check_images
            else bool(not check_root.exists())
        ),
    }
    if not is_full:
        checks.update(
            {
                "F50右边界按形状外扩到可信低点": f50_shape_ok,
                "F11629右边界未扩入外侧噪声": f11629_noise_guard_ok,
                "F1209识别为Seed分裂且共用谷底": f1209_split_ok,
                "F745平坦Seed判0且高峰作为额外真峰": f745_false_ok,
                "F345远端擦边峰不误算Seed分裂": f345_extra_ok,
                "F514额外OUT_FIG与Seed单峰分开": f514_extra_ok,
            }
        )
    if not all(checks.values()):
        raise RuntimeError(f"基础自检失败：{checks}")

    config = {
        "运行模式": "全量训练前数据准备" if is_full else "小批量标注标准化试运行",
        "随机种子": int(args.random_seed),
        "处理范围": (
            "特征表全量稳定顺序"
            if is_full and not selected_sources
            else (
                f"指定mzML来源：{selected_sources}"
                if is_full
                else "分层随机：20张OUT_FIG、40张普通多峰、60张普通单峰、30张假峰；"
                "其中frag1批次各10张多峰/单峰/假峰"
            )
        ),
        "处理来源文件": sorted(sample["source_file"].astype(str).unique()),
        "EIC方法": "nearest",
        "容差策略": tolerance_policy,
        "来源容差映射_ppm": tolerance_map,
        "容差配置来源": tolerance_map_source,
        "容差说明": (
            "10与15 ppm数值门检显示部分原10 ppm来源在15 ppm下会改变EIC、边界和属性，"
            "因此正式结果保留经原图回溯验证的来源级容差，不强制统一。"
            if tolerance_policy == "source-map"
            else f"所有来源统一使用{float(args.tolerance_ppm):g} ppm"
        ),
        "窗口分钟": 2.0,
        "图片尺寸": [480, 480],
        "绘图中心回溯模式": mapping_mode,
        "已确认绘图中心缓存": (
            str(Path(args.mapping_center_cache).resolve())
            if str(args.mapping_center_cache).strip()
            else ""
        ),
        "命中已确认绘图中心缓存数量": mapping_cache_hits,
        "是否输出检查图": write_check_images,
        "是否复制PNG": copy_corrected_images,
        "缺失imageData时是否嵌入原PNG": embed_missing_image_data,
        "Y上边界留白_px": float(args.y_top_pad_px),
        "Y下边界留白_px": float(args.y_bottom_pad_px),
        "普通边界最大移动_秒": float(args.max_boundary_move_sec),
        "普通外边界规则": (
            "5点线性加权平滑后，结合低尾门槛、斜率方向、局部低点与后续回升；"
            "1%仅为候选低位门槛下限，不直接决定边界；允许在原人工边界附近收缩或外扩，最多3秒"
        ),
        "多峰内部边界规则": "相邻人工峰共用两峰顶之间的原始EIC最低点；两侧移动均不超过配置上限",
        "Seed判断规则": (
            "峰顶落入原始Seed区间者作为核心峰；额外分裂成员必须与核心峰共享谷底，"
            f"且峰顶越界不超过{float(args.seed_match_tolerance_sec):g}秒；"
            "仅擦到Seed区间或远端峰保留为图内额外真峰；肩峰/复合框单独用Seed自身信号证据判断；"
            "主要对应峰按峰顶RT距原始Seed RT最近选择，最高峰仅另行记录"
        ),
        "检查图样式": (
            "Arial优先、微软雅黑中文回退；原始/修正边界分轨显示，横坐标不做视觉偏移"
            if write_check_images
            else "全量模式不生成逐图检查图"
        ),
        "统计": stats,
        "自检": checks,
        "输出": {
            "原始种子属性表": str(seed_csv),
            "真峰实例属性表": str(peak_csv),
            "原始标注备份": str(backup_root),
            "修正后标注": str(corrected_root),
            "边界检查图": str(check_root) if write_check_images else "",
        },
    }
    qa_json.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PeakTruthLab标注边界标准化（试运行/全量）")
    parser.add_argument(
        "--feature-csv",
        default=str(PROJECT_ROOT / "datasets" / "feature_table_final_15409.csv"),
    )
    parser.add_argument(
        "--images-root",
        default=str(PROJECT_ROOT / "datasets" / "eic_images_flat"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "datasets" / "PeakTruthLab实验数据"),
    )
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--image-dpi", type=int, default=150)
    parser.add_argument("--y-top-pad-px", type=float, default=4.0)
    parser.add_argument("--y-bottom-pad-px", type=float, default=3.0)
    parser.add_argument("--max-boundary-move-sec", type=float, default=3.0)
    parser.add_argument("--seed-match-tolerance-sec", type=float, default=3.0)
    parser.add_argument(
        "--tolerance-policy",
        choices=["source-map", "uniform"],
        default="source-map",
    )
    parser.add_argument("--tolerance-ppm", type=float, default=15.0)
    parser.add_argument(
        "--source-tolerance-map",
        default="",
        help="JSON来源级ppm映射；可为直接对象或包含source_tolerance_ppm字段",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=["auto", "analytic", "fast", "exhaustive"],
        default="auto",
    )
    parser.add_argument(
        "--mapping-center-cache",
        default="",
        help="可选：已人工确认结果CSV，复用原始特征编号对应的绘图中心RT",
    )
    parser.add_argument(
        "--write-check-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--copy-corrected-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--embed-missing-image-data",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--only-source-files",
        default="",
        help="全量模式可限定逗号分隔的mzML来源",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅用于全量模式的验收限量；0表示不限制",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    if args.y_top_pad_px < 0 or args.y_bottom_pad_px < 0:
        raise ValueError("Y轴留白像素必须非负")
    if args.max_boundary_move_sec <= 0:
        raise ValueError("普通边界移动上限必须大于0秒")
    if args.seed_match_tolerance_sec < 0:
        raise ValueError("Seed匹配容差必须不小于0秒")
    if args.tolerance_ppm <= 0:
        raise ValueError("EIC容差必须大于0 ppm")
    if args.progress_every <= 0:
        raise ValueError("进度输出间隔必须大于0")
    if args.limit < 0:
        raise ValueError("处理数量上限不能为负数")
    return args


if __name__ == "__main__":
    run(parse_args())
