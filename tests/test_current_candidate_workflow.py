from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from PeakTruthLab.scripts.convnext.candidate_components import FusionModel
from lipidbench.data import PeakManifestRecord
from lipidbench.data.training_utils import balanced_candidate_subset, move_peak_batch_to_device


@pytest.mark.parametrize("mode", ["image_only", "attr_only", "naive_concat", "gated_fusion"])
def test_current_classifier_accepts_complete_candidate_attributes(mode: str) -> None:
    model = FusionModel(attr_dim=16, out_dim=1, pretrained=False, model_mode=mode).eval()
    with torch.no_grad():
        logits = model(torch.rand(1, 3, 64, 64), torch.zeros(1, 16))
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("filename", [
    "run_rtx4070_fusion_experiment.py",
    "evaluate_rtx4070_locked_main_test.py",
    "evaluate_rtx4070_lodo_heldout.py",
])
def test_current_classification_entries_load_from_other_directory(tmp_path: Path, filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "PeakTruthLab" / "scripts" / "convnext" / filename
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "usage:" in result.stdout
    assert "--lwga" not in result.stdout


def test_candidate_subsets_remain_deterministic_and_cover_labels() -> None:
    records = [PeakManifestRecord.from_mapping({
        "sample_id": f"candidate-{index}", "image_path": "unused.png",
        "boxes": [], "seed_box": [1, 1, 2, 2], "seed_label": index % 2,
        "attributes": [1.0] * 16, "source_file": "synthetic",
        "study_id": "study", "instrument_id": "instrument",
    }) for index in range(10)]
    selected = balanced_candidate_subset(records, 4, seed=42)
    assert selected == balanced_candidate_subset(records, 4, seed=42)
    assert len({record.sample_id for record in selected}) == 4
    assert {record.seed_label for record in selected} == {0, 1}


def test_batch_device_transfer_preserves_metadata_and_missing_mask() -> None:
    batch = {
        "images": [torch.zeros(3, 64, 64)],
        "targets": [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64)}],
        "seed_boxes": [torch.tensor([[1.0, 1.0, 2.0, 2.0]])],
        "attributes": torch.full((1, 16), float("nan")),
        "attribute_masks": torch.zeros(1, 16, dtype=torch.bool),
        "seed_labels": torch.zeros(1), "metadata": [{"sample_id": "candidate"}],
    }
    moved = move_peak_batch_to_device(batch, torch.device("cpu"))
    assert moved["metadata"] == batch["metadata"]
    assert moved["targets"][0]["boxes"].shape == (0, 4)
    assert moved["attributes"].shape == (1, 16)
    assert torch.isnan(moved["attributes"]).all()
    assert not moved["attribute_masks"].any()
