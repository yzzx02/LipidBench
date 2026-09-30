from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from lipidbench import main as cli
from lipidbench.models import PeakMultiTaskRCNN
from lipidbench.utils.feature_table_io import find_feature_table, load_feature_table
from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries


@pytest.mark.parametrize("scans", [1, 2, 3, 4])
def test_short_chromatograms_keep_valid_scan_indices(scans: int) -> None:
    rt = np.linspace(1.0, 1.1, scans)
    result = refine_peak_boundaries(rt, np.ones(scans), float(rt[0]))
    assert 0 <= result.left_idx <= result.apex_idx <= result.right_idx < scans
    assert np.isfinite([result.rtmin, result.rtmax, result.apex_rt]).all()


@pytest.mark.parametrize("rt", [[1.0, 1.0, 1.1], [1.1, 1.0], [1.0, np.nan]])
def test_invalid_rt_axis_is_rejected(rt: list[float]) -> None:
    with pytest.raises(ValueError, match="retention times"):
        refine_peak_boundaries(np.asarray(rt), np.ones(len(rt)), 1.0)


@pytest.mark.parametrize("algorithm", ["msdial", "ms-dial", "ms_dial"])
def test_msdial_result_discovery_accepts_cli_names(tmp_path: Path, algorithm: str) -> None:
    output = tmp_path / "ms_dial"
    output.mkdir()
    table = output / "sample_processed.csv"
    table.write_text("Feature_ID,mz,RT\nF1,100,1\n", encoding="utf-8")
    assert find_feature_table(tmp_path, algorithm) == table
    assert find_feature_table(output, algorithm) == table


def test_msdial_excel_alias_maps_retention_time_columns(monkeypatch) -> None:
    monkeypatch.setattr(pd, "read_excel", lambda path: pd.DataFrame({"Precursor m/z": [100], "RT (min)": [1]}))
    table = load_feature_table(Path("sample.xlsx"), "ms-dial")
    assert {"Feature_ID", "mz", "RT"}.issubset(table.columns)


def test_eic_export_uses_configured_geometry_and_output_from_other_cwd(tmp_path: Path, monkeypatch) -> None:
    import lipidbench.eic.extract_eic_pyopenms as eic

    output = tmp_path / "custom_msdial"
    output.mkdir()
    (output / "sample_processed.csv").write_text("Feature_ID,mz,RT\nF1,100,1\n", encoding="utf-8")
    config = {
        "paths": {"output_dir": "results"},
        "parameters": {"msdial": {"output_dir": str(output)}},
        "eic_export": {"enabled": True, "mzml": "sample.mzML", "window_min": 3.0,
                       "image_width_px": 640, "image_height_px": 480, "image_dpi": 120},
    }
    args = SimpleNamespace(algo="ms-dial", export_eic=False, eic_mzml=None,
                           **{f"eic_{name}": None for name in
                              ("ppm", "unit", "method", "max_features", "processes", "smooth_sigma")})
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "_get_runner", lambda name: lambda cfg: None)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(eic, "build", lambda paths, table, draw, settings: calls.append((paths, settings)))
    cli.main()
    assert len(calls) == 1
    paths, settings = calls[0]
    assert (settings.window_min, settings.image_width_px, settings.image_height_px, settings.image_dpi) == (3.0, 640, 480, 120)
    project_root = Path(cli.__file__).resolve().parents[1]
    assert paths == [project_root / "sample.mzML"]
    assert Path(settings.images_path) == project_root / "results" / "eic_export"


def test_rpn_rejects_unequal_anchor_counts_before_model_construction() -> None:
    with pytest.raises(ValueError, match="same number of anchors"):
        PeakMultiTaskRCNN(pretrained=False,
                          anchor_sizes=((8,), (16, 24), (32,), (64,)),
                          anchor_aspect_ratios=((1.0,),) * 4)


def test_seed_roi_accepts_float64_external_coordinates() -> None:
    model = PeakMultiTaskRCNN(
        pretrained=False, attr_dim=16, image_min_size=64, image_max_size=64,
        anchor_sizes=((8,), (16,), (32,), (64,)),
        anchor_aspect_ratios=((1.0,),) * 4,
        rpn_pre_nms_top_n_test=8, rpn_post_nms_top_n_test=4,
    ).eval()
    with torch.no_grad():
        output = model([torch.rand(3, 64, 64)],
                       seed_boxes=[torch.tensor([[8, 4, 20, 60]], dtype=torch.float64)],
                       attributes=torch.zeros(1, 16))
    assert torch.isfinite(output["seed_probabilities"]).all()


@pytest.mark.parametrize("filename", ["run_rtx4070_multitask_fusion_experiment.py", "evaluate_rtx4070_multitask_final.py"])
def test_maintained_entry_points_find_repository_from_other_cwd(tmp_path: Path, filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "PeakTruthLab" / "scripts" / "detection" / filename
    environment = {k: v for k, v in os.environ.items()
                   if k not in {"CHROMAPEAK_PROJECT_ROOT", "LIPIDBENCH_PROJECT_ROOT"}}
    code = f"import runpy; print(runpy.run_path({str(script)!r})['PROJECT_ROOT'])"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, check=True)
    assert Path(result.stdout.strip()) == root


def test_current_trainer_uses_manifest_attribute_dimension(tmp_path: Path, monkeypatch) -> None:
    import importlib
    import yaml
    from lipidbench.data import PeakManifestRecord

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "PeakTruthLab" / "scripts" / "detection"))
    trainer = importlib.import_module("run_rtx4070_multitask_fusion_experiment")
    config = yaml.safe_load((root / "PeakTruthLab" / "configs" / "peak_multitask.yaml").read_text(encoding="utf-8"))
    config["model"].pop("attr_dim")
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    record = PeakManifestRecord.from_mapping({
        "sample_id": "synthetic", "image_path": "unused.png", "boxes": [],
        "seed_box": [1, 1, 2, 2], "seed_label": 1, "attributes": [1.0] * 16,
        "source_file": "synthetic", "study_id": "study", "instrument_id": "instrument",
    })
    args = SimpleNamespace(config=config_path, fusion_mode="naive_concat", image_size=96, rpn_train_proposals=8,
                           rpn_test_proposals=4, box_nms_thresh=0.6)
    assert trainer._build_model_config(args, [record])["model"]["attr_dim"] == 16
