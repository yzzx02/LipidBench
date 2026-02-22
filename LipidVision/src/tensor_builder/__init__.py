"""张量构建模块。"""

from .rgb_encoder import build_rgb_tensor
from .tensor_encoder import build_hyperspectral_tensor

__all__ = ["build_rgb_tensor", "build_hyperspectral_tensor"]
