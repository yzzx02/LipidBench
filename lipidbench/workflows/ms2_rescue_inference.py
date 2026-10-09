"""Released-checkpoint adapter for the two-pass MS2 rescue workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lipidbench.data.attribute_preprocessing import AttributePreprocessor
from lipidbench.data.peak_dataset import load_rgb_image_tensor
from lipidbench.data.peak_manifest import BASE_ATTRIBUTE_NAMES
from lipidbench.models import PeakMultiTaskRCNN


class TorchRescuePredictor:
    def __init__(self, detection_checkpoint: str | Path, classification_checkpoint: str | Path,
                 preprocessor_path: str | Path, *, device: str = "auto", batch_size: int = 8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.batch_size = batch_size
        self.preprocessor = AttributePreprocessor.load_json(Path(preprocessor_path))
        if self.preprocessor.attribute_names != BASE_ATTRIBUTE_NAMES:
            raise ValueError("preprocessor must use the current 16 attributes in their canonical order")
        self.detection_model = self._load(Path(detection_checkpoint))
        self.detection_model.detector.roi_heads.score_thresh = 0.0
        self.classification_model = (self.detection_model if Path(detection_checkpoint).resolve() == Path(classification_checkpoint).resolve()
                                     else self._load(Path(classification_checkpoint)))
        self.provenance = {"detection_checkpoint": str(Path(detection_checkpoint).resolve()),
                           "classification_checkpoint": str(Path(classification_checkpoint).resolve()),
                           "attribute_preprocessing": str(Path(preprocessor_path).resolve()), "device": str(self.device)}

    def _load(self, path: Path) -> PeakMultiTaskRCNN:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = PeakMultiTaskRCNN.from_config(checkpoint["config"], pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(self.device).eval()

    @torch.inference_mode()
    def detect(self, image_path: Path) -> list[dict]:
        image = load_rgb_image_tensor(image_path).to(self.device)
        # Only the image detector runs here: no placeholder candidate attributes.
        detection = self.detection_model.detector([image])[0]
        return [{"box": box, "score": score, "label": label}
                for box, score, label in zip(detection["boxes"].cpu().tolist(),
                    detection["scores"].cpu().tolist(), detection["labels"].cpu().tolist(), strict=True)]

    @torch.inference_mode()
    def classify(self, image_path: Path, boxes: list[list[float]], attributes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty(0)
        if np.asarray(attributes).shape != (len(boxes), 16):
            raise ValueError("every candidate must have its own 16-attribute vector")
        image = load_rgb_image_tensor(image_path).to(self.device)
        values = torch.as_tensor(attributes, dtype=torch.float32)
        values = self.preprocessor.transform(values, torch.isfinite(values)).to(self.device)
        probabilities = []
        for start in range(0, len(boxes), self.batch_size):
            chunk = boxes[start:start + self.batch_size]
            # One new candidate per image is the existing public model interface.
            output = self.classification_model([image] * len(chunk),
                seed_boxes=[torch.tensor([box], dtype=torch.float32, device=self.device) for box in chunk],
                attributes=values[start:start + len(chunk)])
            probabilities.extend(output["seed_probabilities"].cpu().tolist())
        return np.asarray(probabilities, dtype=float)
