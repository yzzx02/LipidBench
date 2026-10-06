"""Historical RT inference helpers, using maintained ChromaPeak modules."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import runtime
from lipidbench.utils.peak_attributes import _extract_trace,load_ms1_spectra
from lipidbench.utils.plot_eic import plot_eic
from PeakTruthLab.scripts.annotation.run_annotation_standardization_pilot import _seed_eic_diagnostics
from PeakTruthLab.scripts.detection.evaluate_rtx4070_multitask_final import load_model
SCORE_THRESHOLD=.5

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_and_mapping(rt: np.ndarray, eic: np.ndarray, input_rt: float,
                      out_dir: Path, name: str) -> tuple[Path, float, float]:
    # The project's plotter writes its exact data-to-pixel mapping in LabelMe
    # JSON when given RT limits. These harmless limits are derived solely from
    # input_rt; they are not reference bounds and do not alter the PNG.
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_eic(
        rt, eic, name, out_dir,
        xlim=(input_rt - 1.0, input_rt + 1.0),
        width_px=480, height_px=480, dpi=150,
        normalize_y=False,
        rtmin=input_rt - 0.01, rtmax=input_rt + 0.01,
    )
    json_path = out_dir / f"{name}.json"
    annotation = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.unlink()
    x1 = float(annotation["shapes"][0]["points"][0][0])
    x2 = float(annotation["shapes"][0]["points"][1][0])
    if not x2 > x1:
        raise ValueError("Invalid EIC pixel mapping")
    # LabelMe's generated box pads each side by 5% of the 0.02-min width.
    rt1, rt2 = input_rt - 0.011, input_rt + 0.011
    slope = (rt2 - rt1) / (x2 - x1)
    intercept = rt1 - slope * x1
    return out_dir / f"{name}.png", slope, intercept


def valid_candidates(detection: dict, rt: np.ndarray, eic: np.ndarray,
                     full_rt: np.ndarray, full_eic: np.ndarray,
                     input_rt: float, slope: float, intercept: float) -> list[dict]:
    candidates = []
    for box, score, label in zip(
        detection["boxes"].detach().cpu().numpy(),
        detection["scores"].detach().cpu().numpy(),
        detection["labels"].detach().cpu().numpy(),
        strict=True,
    ):
        if int(label) != 1 or float(score) < SCORE_THRESHOLD:
            continue
        left = max(input_rt - 1.0, float(box[0]) * slope + intercept)
        right = min(input_rt + 1.0, float(box[2]) * slope + intercept)
        if right - left < 0.02:
            continue
        inside = np.flatnonzero((rt >= left) & (rt <= right))
        if inside.size < 3:
            continue
        apex_idx = int(inside[np.argmax(eic[inside])])
        if eic[apex_idx] <= 0:
            continue
        apex = float(rt[apex_idx])
        area = _seed_eic_diagnostics(full_rt, full_eic, apex, left, right)["area"]
        if not np.isfinite(area) or area <= 0:
            continue
        candidates.append({
            "pred_score": float(score),
            "pred_apex_rt": apex,
            "pred_left": left,
            "pred_right": right,
            "pred_area": float(area),
        })
    return candidates
