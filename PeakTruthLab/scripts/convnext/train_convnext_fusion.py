from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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


@dataclass
class AttrScaler:
    fill: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]


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
        attr_scaler: AttrScaler | None = None,
        load_image: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.attr_columns = list(attr_columns)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.class_to_id = class_to_id
        self.attr_scaler = attr_scaler
        self.load_image = bool(load_image)

        required = [image_col, label_col, *self.attr_columns]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        self.samples: list[SampleSpec] = []
        for _, row in self.df.iterrows():
            rel = Path(str(row[image_col]))
            if self.load_image:
                img_path = (self.image_root / rel).resolve()
                if not img_path.exists():
                    continue
            else:
                # Attr-only mode does not need disk image loading.
                img_path = rel

            raw_label = row[label_col]
            if self.class_to_id is None:
                label_id = int(raw_label)
            else:
                label_key = str(raw_label)
                if label_key not in self.class_to_id:
                    raise ValueError(f"Unknown class '{label_key}' in {csv_path}")
                label_id = int(self.class_to_id[label_key])

            attr_vals: list[float] = []
            for c in self.attr_columns:
                v = pd.to_numeric(row.get(c), errors="coerce")
                if pd.isna(v):
                    if self.attr_scaler is not None:
                        v = float(self.attr_scaler.fill[c])
                    else:
                        v = 0.0
                v_f = float(v)
                if self.attr_scaler is not None:
                    v_f = (v_f - float(self.attr_scaler.mean[c])) / float(self.attr_scaler.std[c])
                attr_vals.append(v_f)

            attrs = torch.tensor(attr_vals, dtype=torch.float32)
            self.samples.append(SampleSpec(image_path=img_path, label=label_id, attrs=attrs))

        if not self.samples:
            raise RuntimeError(f"No valid samples found from {csv_path} under {image_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        spec = self.samples[idx]
        if self.load_image:
            image = Image.open(spec.image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros((3, 1, 1), dtype=torch.float32)
        return image, spec.attrs, torch.tensor(spec.label, dtype=torch.long)


class LWGAAttention(nn.Module):
    """Lightweight grouped attention over spatial tokens for long-range dependency modeling."""

    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")

        self.groups = groups
        self.group_dim = channels // groups
        self.scale = self.group_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, groups=groups, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        qkv = self.qkv(x).view(b, 3, self.groups, self.group_dim, n)
        q = qkv[:, 0].permute(0, 1, 3, 2)  # [B, G, N, D]
        k = qkv[:, 1].permute(0, 1, 3, 2)  # [B, G, N, D]
        v = qkv[:, 2].permute(0, 1, 3, 2)  # [B, G, N, D]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).permute(0, 1, 3, 2).contiguous()
        out = out.view(b, c, h, w)
        out = self.proj_drop(self.proj(out))
        return out


class LWGABlock(nn.Module):
    """PreNorm LWGA block with depth-wise FFN branch."""

    def __init__(self, channels: int, groups: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = LWGAAttention(channels=channels, groups=groups, dropout=dropout)

        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionBackbone(nn.Module):
    """ConvNeXt-Tiny backbone with optional LWGA enhancement on final feature map."""

    def __init__(
        self,
        vision_backbone: str = "convnext_tiny",
        pretrained: bool = True,
        lwga_depth: int = 2,
        lwga_groups: int = 8,
        lwga_mlp_ratio: float = 2.0,
        lwga_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pretrained:
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            base = convnext_tiny(weights=None)

        self.features = base.features
        self.avgpool = base.avgpool
        self.out_dim = base.classifier[2].in_features
        self.vision_backbone = vision_backbone

        if vision_backbone == "convnext_tiny":
            self.lwga_layers = nn.Identity()
        elif vision_backbone == "lwga_convnext":
            self.lwga_layers = nn.Sequential(
                *[
                    LWGABlock(
                        channels=self.out_dim,
                        groups=lwga_groups,
                        mlp_ratio=lwga_mlp_ratio,
                        dropout=lwga_dropout,
                    )
                    for _ in range(lwga_depth)
                ]
            )
        else:
            raise ValueError(f"Unsupported vision_backbone: {vision_backbone}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(image)
        feat_map = self.lwga_layers(feat_map)
        pooled = self.avgpool(feat_map)
        return torch.flatten(pooled, 1)


class FusionModel(nn.Module):
    def __init__(
        self,
        attr_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        pretrained: bool = True,
        vision_backbone: str = "convnext_tiny",
        lwga_depth: int = 2,
        lwga_groups: int = 8,
        lwga_mlp_ratio: float = 2.0,
        lwga_dropout: float = 0.0,
        model_mode: str = "gated_fusion",
    ) -> None:
        super().__init__()
        mode = str(model_mode).strip().lower()
        valid_modes = {"image_only", "attr_only", "naive_concat", "gated_fusion"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported model_mode={model_mode}, choose from {sorted(valid_modes)}")
        self.model_mode = mode

        if self.model_mode != "attr_only":
            self.backbone: VisionBackbone | None = VisionBackbone(
                vision_backbone=vision_backbone,
                pretrained=pretrained,
                lwga_depth=lwga_depth,
                lwga_groups=lwga_groups,
                lwga_mlp_ratio=lwga_mlp_ratio,
                lwga_dropout=lwga_dropout,
            )
            in_dim = self.backbone.out_dim
        else:
            self.backbone = None
            in_dim = 0

        if self.model_mode != "image_only":
            self.attr_encoder: nn.Sequential | None = nn.Sequential(
                nn.Linear(attr_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.GELU(),
            )
        else:
            self.attr_encoder = None

        if self.model_mode == "gated_fusion":
            self.gate: nn.Sequential | None = nn.Sequential(
                nn.Linear(in_dim + 64, 256),
                nn.GELU(),
                nn.Linear(256, in_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        if self.model_mode == "image_only":
            classifier_in = in_dim
        elif self.model_mode == "attr_only":
            classifier_in = 64
        else:
            classifier_in = in_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        if self.model_mode == "image_only":
            if self.backbone is None:
                raise RuntimeError("image_only mode requires backbone")
            img_feat = self.backbone(image)
            return self.classifier(img_feat)

        if self.model_mode == "attr_only":
            if self.attr_encoder is None:
                raise RuntimeError("attr_only mode requires attr_encoder")
            attr_feat = self.attr_encoder(attrs)
            return self.classifier(attr_feat)

        if self.backbone is None or self.attr_encoder is None:
            raise RuntimeError(f"mode={self.model_mode} requires both image and attr branches")

        img_feat = self.backbone(image)
        attr_feat = self.attr_encoder(attrs)

        if self.model_mode == "naive_concat":
            final_feat = torch.cat([img_feat, attr_feat], dim=1)
            return self.classifier(final_feat)

        if self.gate is None:
            raise RuntimeError("gated_fusion mode requires gate")
        fused = torch.cat([img_feat, attr_feat], dim=1)
        gate = self.gate(fused)
        gated_img_feat = img_feat * gate
        final_feat = torch.cat([gated_img_feat, attr_feat], dim=1)
        return self.classifier(final_feat)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_attr_scaler_from_train_csv(train_csv: Path, attr_columns: Sequence[str]) -> AttrScaler:
    df = pd.read_csv(train_csv)
    missing = [c for c in attr_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing attribute columns in {train_csv}: {missing}")

    fill: dict[str, float] = {}
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for c in attr_columns:
        s = pd.to_numeric(df[c], errors="coerce")
        med = float(s.median()) if np.isfinite(s.median()) else 0.0
        s2 = s.fillna(med)
        mu = float(s2.mean())
        sigma = float(s2.std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        fill[c] = med
        mean[c] = mu
        std[c] = sigma
    return AttrScaler(fill=fill, mean=mean, std=std)


def build_train_transform(enable_blur: bool, input_size: int) -> transforms.Compose:
    aug = [
        transforms.RandomResizedCrop((input_size, input_size), scale=(0.92, 1.0), ratio=(1.0, 1.0)),
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


def build_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    loss_meter = 0.0

    with torch.no_grad():
        for images, attrs, labels in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            labels_f = labels.to(device).float()

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, attrs).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, labels_f)
                probs = torch.sigmoid(logits)

            loss_meter += float(loss.item()) * labels.shape[0]
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = torch.tensor(all_labels).numpy()
    y_prob = torch.tensor(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "val_loss": loss_meter / max(len(all_labels), 1),
        "val_auc": auc,
        "val_pr_auc": float(average_precision_score(y_true, y_prob)),
        "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "val_tn": int(tn),
        "val_fp": int(fp),
        "val_fn": int(fn),
        "val_tp": int(tp),
    }


def _evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    loss_meter = 0.0

    with torch.no_grad():
        for images, attrs, labels in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
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
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"AMP enabled: {use_amp}")

    attr_columns = [c.strip() for c in args.attr_columns.split(",") if c.strip()]
    attr_scaler = build_attr_scaler_from_train_csv(Path(args.train_csv), attr_columns) if args.standardize_attrs else None

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
        transform=(
            build_train_transform(enable_blur=args.enable_gaussian_blur, input_size=args.input_size)
            if args.model_mode != "attr_only"
            else None
        ),
        class_to_id=class_to_id,
        attr_scaler=attr_scaler,
        load_image=(args.model_mode != "attr_only"),
    )
    val_ds = PeakFusionDataset(
        csv_path=Path(args.val_csv),
        image_root=Path(args.image_root),
        attr_columns=attr_columns,
        image_col=args.image_col,
        label_col=label_col,
        transform=(build_eval_transform(input_size=args.input_size) if args.model_mode != "attr_only" else None),
        class_to_id=class_to_id,
        attr_scaler=attr_scaler,
        load_image=(args.model_mode != "attr_only"),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FusionModel(
        attr_dim=len(attr_columns),
        out_dim=out_dim,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        vision_backbone=args.vision_backbone,
        lwga_depth=args.lwga_depth,
        lwga_groups=args.lwga_groups,
        lwga_mlp_ratio=args.lwga_mlp_ratio,
        lwga_dropout=args.lwga_dropout,
        model_mode=args.model_mode,
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {args.vision_backbone}, mode={args.model_mode}, params={trainable_params / 1e6:.2f}M, "
        f"lwga_depth={args.lwga_depth}, lwga_groups={args.lwga_groups}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device.type, enabled=use_amp)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if attr_scaler is not None:
        with (save_dir / "attr_scaler.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "fill": attr_scaler.fill,
                    "mean": attr_scaler.mean,
                    "std": attr_scaler.std,
                    "attr_columns": attr_columns,
                },
                f,
                indent=2,
            )
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

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, attrs)
                if args.task_type == "binary":
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())
                else:
                    loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]

        train_loss = running_loss / max(seen, 1)
        if args.task_type == "binary":
            metrics = _evaluate_binary(model, val_loader, device, use_amp=use_amp)
            best_metric_name = "val_auc"
        else:
            metrics = _evaluate_multiclass(model, val_loader, device, use_amp=use_amp)
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
                    "class_to_id": class_to_id,
                    "epoch": epoch,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_score,
                    "attr_scaler": (
                        {
                            "fill": attr_scaler.fill,
                            "mean": attr_scaler.mean,
                            "std": attr_scaler.std,
                        }
                        if attr_scaler is not None
                        else None
                    ),
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
        default=(
            "peak_apex_intensity,peak_area_auc,peak_snr_robust,peak_fwhm_min,"
            "peak_asymmetry_factor_10,peak_tailing_factor_5,peak_rt_skewness,"
            "peak_rt_excess_kurtosis,peak_jaggedness,peak_gaussian_similarity,"
            "peak_local_max_count,peak_mz_error_ppm_at_apex"
        ),
        help="Comma-separated attribute column names",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=int, default=480, help="Square image size for both train/eval transforms")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-standardize-attrs",
        action="store_true",
        help="Disable z-score standardization for tabular attributes",
    )
    p.add_argument("--enable-gaussian-blur", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--vision-backbone",
        type=str,
        default="convnext_tiny",
        choices=["convnext_tiny", "lwga_convnext"],
        help="Image branch backbone: baseline ConvNeXt or ConvNeXt+LWGA",
    )
    p.add_argument("--lwga-depth", type=int, default=2, help="Number of LWGA blocks (only for lwga_convnext)")
    p.add_argument("--lwga-groups", type=int, default=8, help="Grouped attention groups (must divide channel dim)")
    p.add_argument("--lwga-mlp-ratio", type=float, default=2.0, help="LWGA FFN expansion ratio")
    p.add_argument("--lwga-dropout", type=float, default=0.0, help="LWGA attention/FFN dropout")
    p.add_argument("--save-dir", type=str, default="PeakTruthLab/models/convnext_fusion")
    p.add_argument(
        "--model-mode",
        type=str,
        default="gated_fusion",
        choices=["image_only", "attr_only", "naive_concat", "gated_fusion"],
        help="Ablation mode for multimodal fusion",
    )
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    args = p.parse_args()
    args.standardize_attrs = not bool(args.no_standardize_attrs)

    if args.lwga_depth < 1:
        raise ValueError("--lwga-depth must be >= 1")
    if args.lwga_groups < 1:
        raise ValueError("--lwga-groups must be >= 1")
    if args.lwga_mlp_ratio <= 0:
        raise ValueError("--lwga-mlp-ratio must be > 0")
    if args.lwga_dropout < 0 or args.lwga_dropout >= 1:
        raise ValueError("--lwga-dropout must be in [0, 1)")
    if args.input_size < 64:
        raise ValueError("--input-size must be >= 64")

    return args


if __name__ == "__main__":
    train(parse_args())
