from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from lipidbench.utils.peak_attributes import Spectrum1, PEAK_ATTRIBUTE_COLUMNS
from lipidbench.utils.plot_eic import plot_eic
from lipidbench.workflows import ms2_rescue as rescue


def _table(records=None):
    return pd.DataFrame(records or [{"target_id": "lipid_A", "sample_id": "sample_1",
        "mzml_path": "sample_1.mzML", "precursor_mz": 760.5851, "ms2_rt": 6.0,
        "has_ms1": False, "has_ms2": True}])


def _spectra(zero=False):
    rt = np.linspace(5.0, 7.0, 1001)
    y = np.zeros_like(rt) if zero else 1000 * np.exp(-0.5 * ((rt - 6) / 0.025) ** 2)
    return [Spectrum1(float(t), np.asarray([760.5851]), np.asarray([v])) for t, v in zip(rt, y, strict=True)]


class Predictor:
    def __init__(self, monkeypatch, probabilities=0.9, interval=(5.94, 6.06), no_detection=False):
        self.probabilities = probabilities
        self.interval = interval
        self.no_detection = no_detection
        self.calls = []
        self.classifications = []
        original = rescue.plot_eic

        def capture(*args, **kwargs):
            geometry = original(*args, **kwargs)
            self.geometry = geometry
            return geometry

        monkeypatch.setattr(rescue, "plot_eic", capture)

    def detect(self, path):
        self.calls.append("detect")
        if self.no_detection:
            return []
        return [{"box": self.geometry.candidate_box(*self.interval, 1000), "score": 0.95}]

    def classify(self, path, boxes, attributes):
        self.calls.append("classify")
        self.classifications.append((boxes, attributes.copy()))
        return np.full(len(boxes), self.probabilities)


def _run(tmp_path, predictor, table=None, **kwargs):
    options = kwargs.pop("options", rescue.RescueOptions(0.5, 0.5))
    return rescue.rescue_ms2_peaks(_table() if table is None else table, predictor, tmp_path / "out",
        options, base_dir=tmp_path, spectra_loader=kwargs.pop("spectra_loader", lambda path: _spectra()), **kwargs)


def test_ms2_only_target_is_detected_then_classified_with_new_roi_and_raw_attributes(tmp_path, monkeypatch):
    predictor = Predictor(monkeypatch)
    result = _run(tmp_path, predictor)
    assert predictor.calls == ["detect", "classify"]
    assert result.targets.status.tolist() == ["rescued"]
    assert result.summary["unique_rescued_peaks"] == 1
    candidate = result.candidates.iloc[0]
    boxes, attributes = predictor.classifications[0]
    assert len(boxes) == 1 and attributes.shape == (1, 16)
    np.testing.assert_allclose(attributes[0], candidate[list(PEAK_ATTRIBUTE_COLUMNS)].to_numpy(dtype=float), equal_nan=True)
    assert boxes[0] == json.loads(candidate.candidate_box)
    assert candidate.detection_box != candidate.candidate_box
    # Independent Gaussian integral checks that the area uses raw RT, not pixels.
    sigma = 0.025
    expected = 1000 * sigma * np.sqrt(np.pi / 2) * (erf((candidate.rtmax - 6) / (np.sqrt(2) * sigma))
                                                  - erf((candidate.rtmin - 6) / (np.sqrt(2) * sigma)))
    assert candidate.area_intensity_min == pytest.approx(expected, rel=5e-4)
    assert candidate.rtmin <= candidate.apex_rt <= candidate.rtmax
    assert (tmp_path / "out/images/target_000001.png").is_file()
    assert len(pd.read_csv(tmp_path / "out/rescued_peaks.csv")) == 1


def test_existing_features_are_reviewed_and_repeated_ms2_annotations_count_once(tmp_path, monkeypatch):
    original = _table().iloc[0].to_dict()
    table = _table([original, {**original, "target_id": "lipid_A_scan2", "ms2_rt": 6.01},
                    {**original, "sample_id": "sample_2", "mzml_path": "sample_2.mzML", "has_ms1": True}])
    loads = []

    def loader(path):
        loads.append(path)
        return _spectra()

    result = _run(tmp_path, Predictor(monkeypatch), table, spectra_loader=loader)
    assert len(loads) == 2  # Load each raw file once, independently of annotation count.
    assert set(result.targets.status) == {"rescued", "duplicate_peak", "existing_validated"}
    assert result.summary["unique_rescued_peaks"] == 1
    assert result.summary["unique_existing_validated_peaks"] == 1
    assert result.summary["rescued_target_rows"] == 2
    assert result.peaks.iloc[0].linked_target_rows == 2


def test_peak_already_present_in_another_annotation_is_not_reported_as_rescued(tmp_path, monkeypatch):
    original = _table().iloc[0].to_dict()
    result = _run(tmp_path, Predictor(monkeypatch), _table([
        original, {**original, "target_id": "existing_A", "has_ms1": True}]))
    assert result.targets.status.tolist() == ["already_present", "existing_validated"]
    assert result.summary["unique_rescued_peaks"] == 0
    assert pd.read_csv(tmp_path / "out/rescued_peaks.csv").empty


@pytest.mark.parametrize("mode,status", [("rejected", "rejected"), ("empty", "no_detection"),
                                         ("remote", "no_valid_candidate"), ("zero", "no_ms1_signal")])
def test_missing_signal_and_failed_candidates_are_not_recovered(tmp_path, monkeypatch, mode, status):
    predictor = Predictor(monkeypatch, probabilities=0.1 if mode == "rejected" else 0.9,
                          interval=(6.5, 6.65) if mode == "remote" else (5.94, 6.06), no_detection=mode == "empty")
    result = _run(tmp_path, predictor, options=rescue.RescueOptions(0.5, 0.5, ms2_rt_tolerance_min=0.02),
                  spectra_loader=lambda path: _spectra(zero=mode == "zero"))
    assert result.targets.status.tolist() == [status]
    assert result.summary["unique_rescued_peaks"] == 0
    assert result.peaks.empty
    if mode in {"empty", "remote", "zero"}:
        assert "classify" not in predictor.calls


def test_rt_seconds_column_mapping_and_relative_raw_paths(tmp_path, monkeypatch):
    table = _table().rename(columns={"ms2_rt": "MS2时间", "precursor_mz": "前体质荷比"})
    table["MS2时间"] = 360.0
    paths = []

    def loader(path):
        paths.append(path)
        return _spectra()

    result = _run(tmp_path, Predictor(monkeypatch), table,
                  options=rescue.RescueOptions(0.5, 0.5, rt_unit="sec"), spectra_loader=loader,
                  column_map={"ms2_rt": "MS2时间", "precursor_mz": "前体质荷比"})
    assert result.targets.ms2_rt_min.tolist() == [6.0]
    assert paths == [tmp_path / "sample_1.mzML"]
    assert result.summary["unique_rescued_peaks"] == 1


def test_inconsistent_sample_file_or_unknown_presence_flag_fails_before_inference(tmp_path, monkeypatch):
    predictor = Predictor(monkeypatch)
    original = _table().iloc[0].to_dict()
    with pytest.raises(ValueError, match="more than one mzML"):
        _run(tmp_path, predictor, _table([original, {**original, "mzml_path": "other.mzML"}]))
    with pytest.raises(ValueError, match="has_ms1"):
        _run(tmp_path, predictor, _table([{**original, "has_ms1": "unknown"}]))
    assert not predictor.calls


def test_loader_errors_and_invalid_classifier_outputs_are_retained(tmp_path, monkeypatch):
    def broken_loader(path):
        raise FileNotFoundError("missing raw file")

    result = _run(tmp_path, Predictor(monkeypatch), spectra_loader=broken_loader)
    assert result.targets.status.tolist() == ["error"]
    assert "missing raw file" in result.targets.iloc[0].error
    bad = Predictor(monkeypatch, probabilities=np.nan)
    result = rescue.rescue_ms2_peaks(_table(), bad, tmp_path / "bad", rescue.RescueOptions(0.5, 0.5),
                                    spectra_loader=lambda path: _spectra())
    assert result.targets.status.tolist() == ["error"]
    assert result.summary["unique_rescued_peaks"] == 0


def test_empty_input_skipped_ms2_and_output_protection(tmp_path, monkeypatch):
    predictor = Predictor(monkeypatch)
    table = _table()
    table["has_ms2"] = False
    table["ms2_rt"] = np.nan
    result = _run(tmp_path, predictor, table)
    assert result.targets.status.tolist() == ["skipped_no_ms2"]
    assert not predictor.calls
    with pytest.raises(FileExistsError):
        _run(tmp_path, predictor)
    result = rescue.rescue_ms2_peaks(_table().iloc[:0], predictor, tmp_path / "empty",
                                    rescue.RescueOptions(0.5, 0.5))
    assert result.summary["input_target_rows"] == 0
    assert result.peaks.empty


def test_image_mapping_uses_axes_and_matches_training_label_padding(tmp_path):
    rt = np.linspace(5, 7, 101)
    intensity = 1000 * np.exp(-0.5 * ((rt - 6) / 0.025) ** 2)
    geometry = plot_eic(rt, intensity, "geometry", tmp_path, xlim=(5, 7), rtmin=5.94, rtmax=6.06)
    expected = json.loads((tmp_path / "geometry.json").read_text())['shapes'][0]['points']
    box = geometry.candidate_box(5.94, 6.06, float(intensity.max()))
    np.testing.assert_allclose(box, np.asarray(expected).reshape(-1), atol=1e-8)
    assert geometry.box_to_rt(box) == pytest.approx((5.934, 6.066))
    assert geometry.axes_left > 0 and geometry.axes_right < 480
    with pytest.raises(ValueError, match="does not overlap"):
        geometry.box_to_rt([geometry.axes_left, 0, geometry.axes_right, geometry.axes_top - 1])


def test_excel_and_delimited_table_intake_preserve_ms1_presence_and_target_mass(tmp_path):
    table = _table()
    for name in ("targets.csv", "targets.tsv", "targets.xlsx"):
        path = tmp_path / name
        if path.suffix == ".xlsx":
            table.to_excel(path, index=False)
        else:
            table.to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",", encoding="utf-8-sig")
        loaded = rescue.read_identification_table(path)
        assert loaded.has_ms1.tolist() == [False]
        assert loaded.precursor_mz.tolist() == pytest.approx([760.5851])


def test_torch_adapter_detects_without_seed_and_classifies_new_candidate_batches(tmp_path, monkeypatch):
    import torch
    from lipidbench.data.attribute_preprocessing import AttributePreprocessor
    from lipidbench.data.peak_dataset import load_rgb_image_tensor
    from lipidbench.data.peak_manifest import BASE_ATTRIBUTE_NAMES
    from lipidbench.models import PeakMultiTaskRCNN
    from lipidbench.workflows.ms2_rescue_inference import TorchRescuePredictor

    model = PeakMultiTaskRCNN(pretrained=False, anchor_sizes=((8,), (16,), (32,), (64,)),
        anchor_aspect_ratios=((1.0, 2.0),) * 4, image_min_size=64, image_max_size=64,
        rpn_pre_nms_top_n_test=8, rpn_post_nms_top_n_test=4, box_detections_per_img=2).eval()
    preprocessing = AttributePreprocessor(BASE_ATTRIBUTE_NAMES, (0.0,) * 16, (1.0,) * 16, (2.0,) * 16, (0,) * 16, 10)
    prep_path = tmp_path / "preprocessing.json"
    preprocessing.save_json(prep_path)
    monkeypatch.setattr(TorchRescuePredictor, "_load", lambda self, path: model)
    adapter = TorchRescuePredictor(tmp_path / "detection.pt", tmp_path / "classification.pt", prep_path,
                                   device="cpu", batch_size=1)
    plot_eic(np.linspace(5, 7, 101), np.ones(101), "input", tmp_path, width_px=64, height_px=64)
    image = tmp_path / "input.png"
    head_calls = []
    hook = model.seed_fusion_head.register_forward_hook(lambda *args: head_calls.append(1))
    adapter.detect(image)
    assert not head_calls
    boxes = [[10, 10, 20, 50], [35, 5, 50, 55]]
    attributes = np.ones((2, 16))
    attributes[0, 0] = np.nan
    attributes[1, 1] = 9.0
    actual = adapter.classify(image, boxes, attributes)
    assert len(head_calls) == 2
    hook.remove()
    values = torch.as_tensor(attributes, dtype=torch.float32)
    with torch.inference_mode():
        expected = model([load_rgb_image_tensor(image)] * 2,
            seed_boxes=[torch.tensor([box], dtype=torch.float32) for box in boxes],
            attributes=preprocessing.transform(values))["seed_probabilities"].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
