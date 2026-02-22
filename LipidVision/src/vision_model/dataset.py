"""高光谱张量数据集读取器。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class HyperspectralTensorDataset(Dataset):
    """读取 `.npy` 高光谱张量并转换为 PyTorch 输入格式。

    输入 `.npy` 约定形状为 (H, W, C)，输出为 (C, H, W) 的 `torch.FloatTensor`。

    Args:
        tensor_dir: `.npy` 张量目录。
        expected_in_channels: 可选，若提供则执行通道一致性校验。
        strict_channel_check: 是否在初始化时扫描并校验所有样本通道数。

    Raises:
        FileNotFoundError: 当目录不存在时抛出。
        ValueError: 当目录中没有 `.npy` 文件或通道不一致时抛出。
    """

    def __init__(
        self,
        tensor_dir: str,
        expected_in_channels: Optional[int] = None,
        strict_channel_check: bool = True,
        file_pattern: str = "*.npy",
    ) -> None:
        super().__init__()
        self.tensor_dir = Path(tensor_dir)
        if not self.tensor_dir.exists():
            raise FileNotFoundError(f"未找到张量目录: {self.tensor_dir}")

        self.file_paths: List[Path] = sorted(self.tensor_dir.glob(file_pattern))
        if len(self.file_paths) == 0:
            raise ValueError(
                f"目录中未找到匹配文件（pattern={file_pattern}）: {self.tensor_dir}"
            )

        self.expected_in_channels = expected_in_channels
        self._discovered_in_channels = self._read_channel_count(self.file_paths[0])

        if self.expected_in_channels is not None and self._discovered_in_channels != int(
            self.expected_in_channels
        ):
            raise ValueError(
                "通道数不匹配："
                f"样本通道={self._discovered_in_channels}, "
                f"expected_in_channels={self.expected_in_channels}"
            )

        if strict_channel_check:
            for path in self.file_paths[1:]:
                ch = self._read_channel_count(path)
                if ch != self._discovered_in_channels:
                    raise ValueError(
                        "检测到动态通道混杂，当前数据集不允许混合通道训练："
                        f"{path.name} 通道数={ch}, "
                        f"首样本通道数={self._discovered_in_channels}"
                    )

    @staticmethod
    def _to_chw_float32(arr: np.ndarray, path: Path) -> np.ndarray:
        """将输入数组统一转换为 (C, H, W) float32。"""
        if arr.ndim != 3:
            raise ValueError(f"无效张量维度（需为 3 维）: {path}")

        # Case 1: (H, W, C)
        if arr.shape[0] == 128 and arr.shape[1] == 128:
            chw = np.transpose(arr, (2, 0, 1))
            return chw.astype(np.float32, copy=False)

        # Case 2: (C, H, W)
        if arr.shape[1] == 128 and arr.shape[2] == 128:
            return arr.astype(np.float32, copy=False)

        raise ValueError(
            f"无法识别的张量布局（期望 HWC 或 CHW，且空间尺寸为 128x128）: {path}"
        )

    @staticmethod
    def _read_channel_count(path: Path) -> int:
        """读取单个文件的通道数。"""
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"无效张量维度（需为 3 维）: {path}")

        # HWC
        if arr.shape[0] == 128 and arr.shape[1] == 128:
            return int(arr.shape[2])

        # CHW
        if arr.shape[1] == 128 and arr.shape[2] == 128:
            return int(arr.shape[0])

        raise ValueError(
            f"无法识别的张量布局（期望 HWC 或 CHW，且空间尺寸为 128x128）: {path}"
        )

    @property
    def in_channels(self) -> int:
        """返回当前数据集检测到的输入通道数。"""
        return int(self._discovered_in_channels)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, str]]:
        """按索引读取样本。

        Args:
            index: 样本索引。

        Returns:
            (tensor, meta):
                - tensor: 形状 (C, H, W) 的 `torch.FloatTensor`
                - meta: 包含 `file_name` 与 `file_path` 的字典
        """
        path = self.file_paths[index]
        arr = np.load(path)
        chw = self._to_chw_float32(arr, path)
        tensor = torch.from_numpy(chw)

        if self.expected_in_channels is not None and tensor.shape[0] != int(
            self.expected_in_channels
        ):
            raise ValueError(
                "运行时通道数不匹配："
                f"{path.name} 通道数={tensor.shape[0]}, "
                f"expected_in_channels={self.expected_in_channels}"
            )

        meta = {
            "file_name": path.name,
            "file_path": str(path),
        }
        return tensor, meta
