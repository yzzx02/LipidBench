from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class LabelMeDetectionDataset(Dataset):
    """Read JPEG + same-name LabelMe JSON for object detection.

    Negative sample support (critical):
    - If JSON has empty shapes [], returns:
      boxes  -> tensor shape (0, 4)
      labels -> tensor shape (0,)
    This is valid for Faster R-CNN training and avoids DataLoader/collate errors.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        label_map: dict[str, int],
        aliases: dict[str, str] | None = None,
        unknown_label_policy: str = "error",
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self.label_map = dict(label_map)
        self.aliases = dict(aliases or {})
        self.unknown_label_policy = str(unknown_label_policy).strip().lower()
        if self.unknown_label_policy not in {"error", "skip", "fallback_true"}:
            raise ValueError("unknown_label_policy must be one of: error|skip|fallback_true")
        if "True_Peak" not in self.label_map:
            raise ValueError("label_map must include 'True_Peak'")

        exts = {".jpg", ".jpeg", ".png"}
        self.images = sorted([p for p in self.data_root.rglob("*") if p.suffix.lower() in exts])
        if not self.images:
            raise RuntimeError(f"No images found under: {self.data_root}")

    def __len__(self) -> int:
        return len(self.images)

    def _load_labelme_boxes(self, json_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not json_path.exists():
            # Missing JSON -> treat as negative sample
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            return boxes, labels

        with json_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        shapes: list[dict[str, Any]] = ann.get("shapes", []) or []
        if len(shapes) == 0:
            # Negative sample (explicit empty shapes)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            return boxes, labels

        parsed_boxes: list[list[float]] = []
        parsed_labels: list[int] = []

        for s in shapes:
            pts = s.get("points", [])
            if len(pts) < 2:
                continue

            (x1, y1), (x2, y2) = pts[0], pts[1]
            xmin, xmax = float(min(x1, x2)), float(max(x1, x2))
            ymin, ymax = float(min(y1, y2)), float(max(y1, y2))

            # Skip invalid / degenerate boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            parsed_boxes.append([xmin, ymin, xmax, ymax])

            label_name = str(s.get("label", "")).strip()
            if label_name in self.aliases:
                label_name = self.aliases[label_name]

            if label_name in self.label_map:
                parsed_labels.append(int(self.label_map[label_name]))
            else:
                if self.unknown_label_policy == "skip":
                    parsed_boxes.pop()
                    continue
                if self.unknown_label_policy == "fallback_true":
                    parsed_labels.append(int(self.label_map["True_Peak"]))
                else:
                    raise ValueError(f"Unknown label '{label_name}' in {json_path}")

        if len(parsed_boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(parsed_boxes, dtype=torch.float32)
            labels = torch.tensor(parsed_labels, dtype=torch.int64)

        return boxes, labels

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        json_path = img_path.with_suffix(".json")

        image = Image.open(img_path).convert("RGB")
        image_tensor = to_tensor(image).clamp(0.0, 1.0)  # [0,1], robust to rare abnormal values

        boxes, labels = self._load_labelme_boxes(json_path)

        if boxes.numel() == 0:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    # Detection task requires list[Tensor], list[Dict]
    images, targets = tuple(zip(*batch))
    return list(images), list(targets)


def build_model(num_classes: int = 2) -> torch.nn.Module:
    # Keep compatibility with old/new torchvision APIs
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    except Exception:
        # Older torchvision
        model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map, aliases = load_label_map(Path(args.label_map))
    dataset = LabelMeDetectionDataset(
        args.data_root,
        label_map=label_map,
        aliases=aliases,
        unknown_label_policy=args.unknown_label,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    num_classes = int(max(label_map.values())) + 1
    model = build_model(num_classes=num_classes).to(device)
    model.train()

    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = save_path.with_name(f"{save_path.stem}_latest.pth")

    start_epoch = 0
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"Resumed from latest checkpoint: {latest_path} (start_epoch={start_epoch})")
        else:
            model.load_state_dict(ckpt)
            print(f"Loaded model-only checkpoint: {latest_path}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        for step, (images, targets) in enumerate(loader, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_items = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
            print(f"  step={step:04d} total_loss={float(total_loss.detach().cpu()):.4f} details={loss_items}")

        # checkpoint each epoch (disaster-safe)
        epoch_path = save_path.with_name(f"{save_path.stem}_epoch_{epoch + 1}.pth")
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_map": label_map,
            "aliases": aliases,
        }
        torch.save(state, epoch_path)
        torch.save(state, latest_path)
        print(f"Checkpoint saved: {epoch_path}")
        print(f"Latest updated:  {latest_path}")

    if args.save_path:
        # keep backward-compatible model-only file for inference scripts
        torch.save(model.state_dict(), save_path)
        print(f"Model saved (state_dict): {save_path}")


def parse_args():
    parser = argparse.ArgumentParser("Minimal Faster R-CNN trainer for PeakTruthLab")
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\datasets\eic_images",
        help="Root folder containing images and same-name LabelMe JSON files",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default=r"D:\LipidBench\PeakTruthLab\models\fasterrcnn_mvp.pth")
    parser.add_argument(
        "--label-map",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.json"),
        help="JSON file defining label->class_id mapping (background is implicit 0)",
    )
    parser.add_argument(
        "--unknown-label",
        type=str,
        default="error",
        choices=["error", "skip", "fallback_true"],
        help="How to handle unknown LabelMe labels",
    )
    return parser.parse_args()


def load_label_map(label_map_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    labels_obj = obj.get("labels") or {}
    aliases_obj = obj.get("aliases") or {}
    label_map = {str(k): int(v) for k, v in labels_obj.items()}
    aliases = {str(k): str(v) for k, v in aliases_obj.items()}
    if not label_map:
        raise ValueError(f"label_map has empty labels: {label_map_path}")
    return label_map, aliases


if __name__ == "__main__":
    args = parse_args()
    train(args)
