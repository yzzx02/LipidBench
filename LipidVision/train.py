"""最小可行训练脚本（Synthetic Pre-training）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.vision_model.dataset import HyperspectralTensorDataset
from src.vision_model.synthetic_generator import generate_synthetic_dataset
from src.vision_model.unet_segmenter import UNetSegmenter


class SegmentationPairDataset(Dataset):
    """将 `HyperspectralTensorDataset` 与同名 mask 文件配对。"""

    def __init__(self, image_dir: str, mask_dir: str, expected_in_channels: int | None = None) -> None:
        super().__init__()
        self.image_ds = HyperspectralTensorDataset(
            tensor_dir=image_dir,
            expected_in_channels=expected_in_channels,
            strict_channel_check=True,
            file_pattern="image_*.npy",
        )
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"未找到 mask 目录: {self.mask_dir}")

    @property
    def in_channels(self) -> int:
        return self.image_ds.in_channels

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, str]]:
        x, meta = self.image_ds[idx]
        image_name = meta["file_name"]
        mask_name = image_name.replace("image_", "mask_", 1)
        mask_path = self.mask_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"未找到配对 mask: {mask_path}")

        mask_arr = np.load(mask_path)
        if mask_arr.ndim != 3:
            raise ValueError(f"mask 维度错误（需 3 维）: {mask_path}")

        # 兼容 (1,H,W) 或 (H,W,1)
        if mask_arr.shape[0] == 1 and mask_arr.shape[1] == 128 and mask_arr.shape[2] == 128:
            mask = torch.from_numpy(mask_arr.astype(np.float32, copy=False))
        elif mask_arr.shape[0] == 128 and mask_arr.shape[1] == 128 and mask_arr.shape[2] == 1:
            chw = np.transpose(mask_arr, (2, 0, 1)).astype(np.float32, copy=False)
            mask = torch.from_numpy(chw)
        else:
            raise ValueError(f"mask 布局不支持（期望 1x128x128 或 128x128x1）: {mask_path}")

        return x, mask, meta


class DiceLoss(nn.Module):
    """二值 Dice Loss。"""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data/synthetic_train")
    if not data_dir.exists() or len(list(data_dir.glob("image_*.npy"))) == 0:
        generate_synthetic_dataset(
            output_dir=str(data_dir),
            n_samples=200,
            width=128,
            height=128,
            channels=5,
            noise_std=0.01,
            seed=2026,
        )

    dataset = SegmentationPairDataset(
        image_dir=str(data_dir),
        mask_dir=str(data_dir),
        expected_in_channels=None,
    )

    model = UNetSegmenter(in_channels=dataset.in_channels, out_channels=1, base_channels=32).to(device)
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y, _meta in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.6f}")


if __name__ == "__main__":
    main()
