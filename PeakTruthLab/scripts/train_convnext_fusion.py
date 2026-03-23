from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


@dataclass
class SampleSpec:
    image_path: Path
    label: int
    attrs: torch.Tensor


class PeakFusionDataset(Dataset):
    """Dataset for image + tabular attributes classification (binary or multiclass)."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        attr_columns: Sequence[str],
        image_col: str,
        label_col: str,
        transform: transforms.Compose | None = None,
        class_to_id: dict[str, int] | None = None,
        attr_mean: torch.Tensor | None = None,
        attr_std: torch.Tensor | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.attr_columns = list(attr_columns)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.class_to_id = class_to_id
        self.attr_mean = attr_mean
        self.attr_std = attr_std

        required = [image_col, label_col, *self.attr_columns]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        self.samples: list[SampleSpec] = []
        for _, row in self.df.iterrows():
            rel = Path(str(row[image_col]))
            img_path = (self.image_root / rel).resolve()
            if not img_path.exists():
                continue

            raw_label = row[label_col]
            if self.class_to_id is None:
                label_id = int(raw_label)
            else:
                label_key = str(raw_label)
                if label_key not in self.class_to_id:
                    raise ValueError(f"Unknown class '{label_key}' in {csv_path}")
                label_id = int(self.class_to_id[label_key])

            attrs = torch.tensor([float(row[c]) for c in self.attr_columns], dtype=torch.float32)
            if self.attr_mean is not None and self.attr_std is not None:
                attrs = (attrs - self.attr_mean) / self.attr_std
            self.samples.append(SampleSpec(image_path=img_path, label=label_id, attrs=attrs))

        if not self.samples:
            raise RuntimeError(f"No valid samples found from {csv_path} under {image_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        spec = self.samples[idx]
        image = Image.open(spec.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, spec.attrs, torch.tensor(spec.label, dtype=torch.long)


class FusionModel(nn.Module):
    def __init__(
        self,
        attr_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            backbone = convnext_tiny(weights=None)

        in_dim = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.attr_encoder = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(in_dim + 64, 256),
            nn.GELU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_dim + 64, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)
        attr_feat = self.attr_encoder(attrs)
        fused = torch.cat([img_feat, attr_feat], dim=1)

        gate = self.gate(fused)
        gated_img_feat = img_feat * gate
        final_feat = torch.cat([gated_img_feat, attr_feat], dim=1)
        logits = self.classifier(final_feat)
        return logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(enable_blur: bool) -> transforms.Compose:
    aug = [
        transforms.RandomResizedCrop((300, 400), scale=(0.92, 1.0), ratio=(400 / 300, 400 / 300)),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
    ]
    if enable_blur:
        aug.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15))

    aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(aug)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((300, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _evaluate_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    loss_meter = 0.0

    with torch.no_grad():
        for images, attrs, labels in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            labels_f = labels.to(device).float()

            logits = model(images, attrs).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels_f)
            probs = torch.sigmoid(logits)

            loss_meter += float(loss.item()) * labels.shape[0]
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = torch.tensor(all_labels).numpy()
    y_prob = torch.tensor(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "val_loss": loss_meter / max(len(all_labels), 1),
        "val_auc": float(roc_auc_score(y_true, y_prob)),
        "val_pr_auc": float(average_precision_score(y_true, y_prob)),
        "val_f1": float(f1_score(y_true, y_pred)),
        "val_acc": float(accuracy_score(y_true, y_pred)),
    }


def _evaluate_multiclass(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    loss_meter = 0.0

    with torch.no_grad():
        for images, attrs, labels in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            logits = model(images, attrs)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)

            loss_meter += float(loss.item()) * labels.shape[0]
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    y_true = torch.tensor(all_labels).numpy()
    y_pred = torch.tensor(all_preds).numpy()

    return {
        "val_loss": loss_meter / max(len(all_labels), 1),
        "val_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "val_acc": float(accuracy_score(y_true, y_pred)),
    }


def _build_class_mapping(train_csv: Path, class_col: str, class_map_json: str | None) -> dict[str, int]:
    if class_map_json:
        with Path(class_map_json).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return {str(k): int(v) for k, v in obj.items()}

    df = pd.read_csv(train_csv)
    if class_col not in df.columns:
        raise ValueError(f"Missing class column '{class_col}' in {train_csv}")
    names = sorted({str(x) for x in df[class_col].dropna().tolist()})
    return {k: i for i, k in enumerate(names)}


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    attr_columns = [c.strip() for c in args.attr_columns.split(",") if c.strip()]

    train_df_for_stats = pd.read_csv(Path(args.train_csv))
    missing_attr = [c for c in attr_columns if c not in train_df_for_stats.columns]
    if missing_attr:
        raise ValueError(f"Missing attr columns in train CSV: {missing_attr}")

    attr_mean = torch.tensor(train_df_for_stats[attr_columns].astype(float).mean(axis=0).values, dtype=torch.float32)
    attr_std = torch.tensor(train_df_for_stats[attr_columns].astype(float).std(axis=0).values, dtype=torch.float32)
    attr_std = torch.where(attr_std < 1e-8, torch.ones_like(attr_std), attr_std)

    print("Attribute normalization enabled (z-score from train split).")

    if args.task_type == "binary":
        label_col = args.label_col
        class_to_id = None
        out_dim = 1
    else:
        label_col = args.class_col
        class_to_id = _build_class_mapping(Path(args.train_csv), args.class_col, args.class_map_json)
        out_dim = len(class_to_id)
        if out_dim < 2:
            raise ValueError("Multiclass mode requires at least 2 classes")
        print(f"Multiclass mapping: {class_to_id}")

    train_ds = PeakFusionDataset(
        csv_path=Path(args.train_csv),
        image_root=Path(args.image_root),
        attr_columns=attr_columns,
        image_col=args.image_col,
        label_col=label_col,
        transform=build_train_transform(enable_blur=args.enable_gaussian_blur),
        class_to_id=class_to_id,
        attr_mean=attr_mean,
        attr_std=attr_std,
    )
    val_ds = PeakFusionDataset(
        csv_path=Path(args.val_csv),
        image_root=Path(args.image_root),
        attr_columns=attr_columns,
        image_col=args.image_col,
        label_col=label_col,
        transform=build_eval_transform(),
        class_to_id=class_to_id,
        attr_mean=attr_mean,
        attr_std=attr_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FusionModel(
        attr_dim=len(attr_columns),
        out_dim=out_dim,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, attrs, labels in train_loader:
            images = images.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            logits = model(images, attrs)
            if args.task_type == "binary":
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]

        train_loss = running_loss / max(seen, 1)
        if args.task_type == "binary":
            metrics = _evaluate_binary(model, val_loader, device)
            best_metric_name = "val_auc"
        else:
            metrics = _evaluate_multiclass(model, val_loader, device)
            best_metric_name = "val_f1_macro"

        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(record)
        print(record)

        score = float(record[best_metric_name])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "attr_columns": attr_columns,
                    "attr_mean": attr_mean.tolist(),
                    "attr_std": attr_std.tolist(),
                    "class_to_id": class_to_id,
                    "epoch": epoch,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_score,
                },
                save_dir / "best_model.pth",
            )

    with (save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training done. Best {best_metric_name}={best_score:.4f}")
    print(f"Saved model/history to: {save_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ConvNeXt-Tiny + Attribute Fusion Trainer")
    p.add_argument("--train-csv", type=str, required=True, help="CSV with train split")
    p.add_argument("--val-csv", type=str, required=True, help="CSV with validation split")
    p.add_argument("--image-root", type=str, required=True, help="Root folder for images")
    p.add_argument("--image-col", type=str, default="image")

    p.add_argument("--task-type", type=str, default="binary", choices=["binary", "multiclass"])
    p.add_argument("--label-col", type=str, default="is_true_peak", help="Binary label column (0/1)")
    p.add_argument("--class-col", type=str, default="peak_class", help="Multiclass label column")
    p.add_argument("--class-map-json", type=str, default=None, help="Optional class mapping JSON for multiclass")

    p.add_argument(
        "--attr-columns",
        type=str,
        default="slope,sharpness,height,sn,width,mass_accuracy",
        help="Comma-separated attribute column names",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-gaussian-blur", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--save-dir", type=str, default="PeakTruthLab/models/convnext_fusion")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
