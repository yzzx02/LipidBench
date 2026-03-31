from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundaryRefinementResult:
    status: str
    rtmin: float
    rtmax: float
    apex_rt: float
    left_idx: int
    right_idx: int
    apex_idx: int
    width_sec: float
    bound_mode: str
    old_rt_in_bounds: bool
    rt_recentred: bool
    baseline: float
    noise_sigma: float
    threshold: float
    af: float
    ff: float
    sf: float
    left_expand_scans: int
    right_expand_scans: int
    left_rebound_stop: bool
    right_rebound_stop: bool
    oversized_shrink: bool


def _sanitize_trace(rt: np.ndarray, eic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rt_arr = np.asarray(rt, dtype=np.float64)
    y = np.asarray(eic, dtype=np.float64)
    if rt_arr.ndim != 1 or y.ndim != 1 or rt_arr.size != y.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if rt_arr.size == 0:
        return rt_arr, y
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    return rt_arr, y


def _make_triangular_kernel(window_scans: int) -> np.ndarray:
    w = max(1, int(window_scans))
    if w % 2 == 0:
        w += 1
    half = w // 2
    left = np.arange(1, half + 2, dtype=np.float64)
    kernel = np.concatenate([left, left[-2::-1]])
    kernel /= np.sum(kernel)
    return kernel


def _lwma_smooth(y: np.ndarray, window_scans: int, passes: int) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    kernel = _make_triangular_kernel(window_scans)
    out = np.asarray(y, dtype=np.float64)
    for _ in range(max(1, int(passes))):
        out = np.convolve(out, kernel, mode="same")
    return out


def _trimmed_baseline_sigma(y: np.ndarray, trim_ratio: float = 0.05) -> tuple[float, float]:
    yy = np.asarray(y, dtype=np.float64)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return 0.0, 0.0
    if yy.size < 8:
        baseline = float(np.median(yy))
        mad = float(np.median(np.abs(yy - baseline)))
        return baseline, float(1.4826 * mad)

    ys = np.sort(yy)
    trim = int(np.floor(float(trim_ratio) * ys.size))
    lo = min(max(trim, 0), max(ys.size - 2, 0))
    hi = max(lo + 1, ys.size - trim)
    core = ys[lo:hi]
    if core.size < 3:
        core = ys

    baseline = float(np.mean(core))
    sigma = float(np.std(core, ddof=0))
    if not np.isfinite(sigma):
        sigma = 0.0
    return baseline, sigma


def _msdial_noise_filters(rt: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    if ys.size < 5:
        return 1e-4, 1e-4, 1e-4

    amp = np.abs(np.diff(ys))
    d1 = np.abs(np.gradient(ys, rt))
    d2 = np.abs(np.gradient(d1, rt))

    def _pick(v: np.ndarray) -> float:
        vv = np.asarray(v, dtype=np.float64)
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            return 1e-4
        vmax = float(np.max(vv))
        if vmax <= 1e-12:
            return 1e-4
        keep = vv[vv <= (0.05 * vmax)]
        if keep.size == 0:
            keep = vv
        med = float(np.median(keep))
        if med <= 1e-12:
            med = 1e-4
        return med

    return _pick(amp), _pick(d1), _pick(d2)


def _nearest_idx(rt: np.ndarray, value: float) -> int:
    if rt.size == 0:
        return -1
    return int(np.argmin(np.abs(rt - float(value))))


def _pick_apex_idx(
    rt: np.ndarray,
    ys: np.ndarray,
    rt_hint: float,
    search_half_window_min: float,
    *,
    allow_far_apex: bool,
) -> int:
    if rt.size == 0:
        return -1
    half = max(float(search_half_window_min), 1e-6)
    rt_step = float(np.median(np.diff(rt))) if rt.size >= 3 else 0.0
    prefer_half = min(half, max(0.05, 6.0 * max(rt_step, 0.0)))
    local_half = min(half, max(0.12, 12.0 * max(rt_step, 0.0)))

    mask_prefer = (rt >= (float(rt_hint) - prefer_half)) & (rt <= (float(rt_hint) + prefer_half))
    if np.any(mask_prefer) and float(np.max(ys[mask_prefer])) > 0:
        cand_idx = np.where(mask_prefer)[0]
        return int(cand_idx[np.argmax(ys[cand_idx])])

    mask_local = (rt >= (float(rt_hint) - local_half)) & (rt <= (float(rt_hint) + local_half))
    if np.any(mask_local) and float(np.max(ys[mask_local])) > 0:
        cand_idx = np.where(mask_local)[0]
        return int(cand_idx[np.argmax(ys[cand_idx])])

    mask = (rt >= (float(rt_hint) - half)) & (rt <= (float(rt_hint) + half))
    if allow_far_apex and np.any(mask) and float(np.max(ys[mask])) > 0:
        cand_idx = np.where(mask)[0]
        return int(cand_idx[np.argmax(ys[cand_idx])])
    return _nearest_idx(rt, float(rt_hint))


def _is_baseline_like(
    ys: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    amp: np.ndarray,
    idx: int,
    *,
    threshold: float,
    af: float,
    ff: float,
    sf: float,
) -> bool:
    if idx < 0 or idx >= ys.size:
        return True
    amp_idx = min(max(idx, 1), amp.size) - 1 if amp.size > 0 else 0
    amp_val = float(amp[amp_idx]) if amp.size > 0 else 0.0
    return bool(
        ys[idx] <= threshold
        and abs(float(d1[idx])) <= max(float(ff) * 1.5, 1e-9)
        and abs(float(d2[idx])) <= max(float(sf) * 1.5, 1e-9)
        and amp_val <= max(float(af) * 1.5, 1e-9)
    )


def _walk_to_core_boundary(
    rt: np.ndarray,
    ys: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    amp: np.ndarray,
    apex_idx: int,
    *,
    direction: int,
    threshold: float,
    af: float,
    ff: float,
    sf: float,
    confirm_scans: int,
    max_walk_scans: int,
) -> int:
    idx = int(apex_idx)
    streak = 0
    steps = 0

    while 0 <= idx + direction < len(rt) and steps < int(max_walk_scans):
        idx += direction
        steps += 1
        if _is_baseline_like(ys, d1, d2, amp, idx, threshold=threshold, af=af, ff=ff, sf=sf):
            streak += 1
            if streak >= max(1, int(confirm_scans)):
                break
        else:
            streak = 0
    return int(idx)


def _extend_to_local_minimum(
    rt: np.ndarray,
    ys: np.ndarray,
    start_idx: int,
    *,
    direction: int,
    threshold: float,
    af: float,
    max_expand_scans: int,
    max_expand_min: float,
    rise_rel_tol: float,
    rebound_rel: float,
    rise_patience: int,
    flat_baseline_patience: int = 3,
) -> tuple[int, bool]:
    if start_idx < 0 or start_idx >= len(rt):
        return int(start_idx), False

    idx = int(start_idx)
    valley_idx = int(start_idx)
    valley_val = float(ys[start_idx])
    start_rt = float(rt[start_idx])
    steps = 0
    rise_streak = 0
    low_flat_streak = 0
    saw_descent = False

    while 0 <= idx + direction < len(rt):
        nxt_idx = idx + direction
        steps += 1
        if steps > int(max_expand_scans):
            break
        if abs(float(rt[nxt_idx]) - start_rt) > float(max_expand_min):
            break

        cur = float(ys[idx])
        nxt = float(ys[nxt_idx])

        if nxt <= cur * (1.0 + float(rise_rel_tol)):
            idx = nxt_idx
            saw_descent = True
            rise_streak = 0
            if nxt < valley_val:
                valley_val = nxt
                valley_idx = idx

            if cur <= threshold and nxt <= threshold and abs(nxt - cur) <= max(float(af), 1e-9):
                low_flat_streak += 1
                if low_flat_streak >= int(max(1, flat_baseline_patience)):
                    break
            else:
                low_flat_streak = 0
            continue

        low_flat_streak = 0
        rise_streak += 1
        if saw_descent and nxt >= valley_val * (1.0 + float(rebound_rel)):
            return int(valley_idx), True
        if rise_streak > int(max(0, rise_patience)):
            break
        idx = nxt_idx

    return int(valley_idx if saw_descent else start_idx), False


def _ensure_order(rt: np.ndarray, left_idx: int, right_idx: int) -> tuple[int, int]:
    li = int(min(left_idx, right_idx))
    ri = int(max(left_idx, right_idx))
    li = max(0, li)
    ri = min(len(rt) - 1, ri)
    return li, ri


def refine_peak_boundaries(
    rt: np.ndarray,
    eic: np.ndarray,
    rt_hint: float,
    *,
    rtmin_hint: float | None = None,
    rtmax_hint: float | None = None,
    search_half_window_min: float = 0.35,
    local_half_window_min: float = 1.0,
    smooth_window_scans: int = 5,
    smooth_passes: int = 2,
    sigma_mult: float = 3.0,
    min_rel_height: float = 0.02,
    confirm_scans: int = 2,
    max_expand_scans: int = 18,
    max_expand_min: float = 0.35,
    rise_rel_tol: float = 0.02,
    rebound_rel: float = 0.12,
    rise_patience: int = 2,
    oversize_factor: float = 1.8,
) -> BoundaryRefinementResult:
    rt_arr, y_raw = _sanitize_trace(rt, eic)
    old_rt_in_bounds = False
    if rtmin_hint is not None and rtmax_hint is not None and np.isfinite(rtmin_hint) and np.isfinite(rtmax_hint):
        lo = float(min(rtmin_hint, rtmax_hint))
        hi = float(max(rtmin_hint, rtmax_hint))
        old_rt_in_bounds = bool(lo <= float(rt_hint) <= hi)

    if rt_arr.size == 0:
        return BoundaryRefinementResult(
            status="empty_trace",
            rtmin=float(rt_hint),
            rtmax=float(rt_hint),
            apex_rt=float(rt_hint),
            left_idx=-1,
            right_idx=-1,
            apex_idx=-1,
            width_sec=0.0,
            bound_mode="empty_trace",
            old_rt_in_bounds=old_rt_in_bounds,
            rt_recentred=False,
            baseline=0.0,
            noise_sigma=0.0,
            threshold=0.0,
            af=1e-4,
            ff=1e-4,
            sf=1e-4,
            left_expand_scans=0,
            right_expand_scans=0,
            left_rebound_stop=False,
            right_rebound_stop=False,
            oversized_shrink=False,
        )

    ys = _lwma_smooth(y_raw, window_scans=smooth_window_scans, passes=smooth_passes)
    amp = np.abs(np.diff(ys))
    d1 = np.gradient(ys, rt_arr) if rt_arr.size >= 3 else np.zeros_like(ys)
    d2 = np.gradient(d1, rt_arr) if rt_arr.size >= 3 else np.zeros_like(ys)
    af, ff, sf = _msdial_noise_filters(rt_arr, ys)

    apex_idx = _pick_apex_idx(
        rt_arr,
        ys,
        float(rt_hint),
        float(search_half_window_min),
        allow_far_apex=not old_rt_in_bounds,
    )
    if apex_idx < 0:
        apex_idx = _nearest_idx(rt_arr, float(rt_hint))
    apex_rt = float(rt_arr[apex_idx])
    apex_int = float(ys[apex_idx]) if apex_idx >= 0 else 0.0

    local_mask = (rt_arr >= (apex_rt - float(local_half_window_min))) & (rt_arr <= (apex_rt + float(local_half_window_min)))
    local_y = ys[local_mask] if np.any(local_mask) else ys
    baseline, noise_sigma = _trimmed_baseline_sigma(local_y)

    dynamic_thr = baseline + max(float(sigma_mult) * noise_sigma, (apex_int - baseline) * float(min_rel_height))
    threshold = float(max(baseline, dynamic_thr))
    if apex_int <= 0:
        return BoundaryRefinementResult(
            status="zero_apex",
            rtmin=apex_rt,
            rtmax=apex_rt,
            apex_rt=apex_rt,
            left_idx=apex_idx,
            right_idx=apex_idx,
            apex_idx=apex_idx,
            width_sec=0.0,
            bound_mode="zero_apex",
            old_rt_in_bounds=old_rt_in_bounds,
            rt_recentred=not old_rt_in_bounds,
            baseline=baseline,
            noise_sigma=noise_sigma,
            threshold=threshold,
            af=af,
            ff=ff,
            sf=sf,
            left_expand_scans=0,
            right_expand_scans=0,
            left_rebound_stop=False,
            right_rebound_stop=False,
            oversized_shrink=False,
        )

    if threshold >= apex_int:
        threshold = float(baseline + max((apex_int - baseline) * 0.35, 1e-9))

    max_walk_scans = max(20, int(max_expand_scans) * 4)
    left_core = _walk_to_core_boundary(
        rt_arr,
        ys,
        d1,
        d2,
        amp,
        apex_idx,
        direction=-1,
        threshold=threshold,
        af=af,
        ff=ff,
        sf=sf,
        confirm_scans=confirm_scans,
        max_walk_scans=max_walk_scans,
    )
    right_core = _walk_to_core_boundary(
        rt_arr,
        ys,
        d1,
        d2,
        amp,
        apex_idx,
        direction=1,
        threshold=threshold,
        af=af,
        ff=ff,
        sf=sf,
        confirm_scans=confirm_scans,
        max_walk_scans=max_walk_scans,
    )
    left_core, right_core = _ensure_order(rt_arr, left_core, right_core)

    left_ext, left_rebound = _extend_to_local_minimum(
        rt_arr,
        ys,
        left_core,
        direction=-1,
        threshold=threshold,
        af=af,
        max_expand_scans=max_expand_scans,
        max_expand_min=max_expand_min,
        rise_rel_tol=rise_rel_tol,
        rebound_rel=rebound_rel,
        rise_patience=rise_patience,
    )
    right_ext, right_rebound = _extend_to_local_minimum(
        rt_arr,
        ys,
        right_core,
        direction=1,
        threshold=threshold,
        af=af,
        max_expand_scans=max_expand_scans,
        max_expand_min=max_expand_min,
        rise_rel_tol=rise_rel_tol,
        rebound_rel=rebound_rel,
        rise_patience=rise_patience,
    )
    left_ext, right_ext = _ensure_order(rt_arr, left_ext, right_ext)

    core_width = float(max(rt_arr[right_core] - rt_arr[left_core], 1e-12))
    ext_width = float(max(rt_arr[right_ext] - rt_arr[left_ext], 0.0))
    edge_low = bool(ys[left_ext] <= threshold and ys[right_ext] <= threshold)
    oversized = bool(edge_low and ext_width > (core_width * float(oversize_factor)))

    if oversized:
        left_final, right_final = left_core, right_core
        bound_mode = "core_shrink"
    else:
        left_final, right_final = left_ext, right_ext
        bound_mode = "valley_extend"

    left_final, right_final = _ensure_order(rt_arr, left_final, right_final)
    rtmin = float(rt_arr[left_final])
    rtmax = float(rt_arr[right_final])

    return BoundaryRefinementResult(
        status="ok",
        rtmin=rtmin,
        rtmax=rtmax,
        apex_rt=apex_rt,
        left_idx=left_final,
        right_idx=right_final,
        apex_idx=apex_idx,
        width_sec=float(max(rtmax - rtmin, 0.0) * 60.0),
        bound_mode=bound_mode,
        old_rt_in_bounds=old_rt_in_bounds,
        rt_recentred=not old_rt_in_bounds,
        baseline=baseline,
        noise_sigma=noise_sigma,
        threshold=threshold,
        af=af,
        ff=ff,
        sf=sf,
        left_expand_scans=int(max(0, left_core - left_ext)),
        right_expand_scans=int(max(0, right_ext - right_core)),
        left_rebound_stop=bool(left_rebound),
        right_rebound_stop=bool(right_rebound),
        oversized_shrink=bool(oversized),
    )
