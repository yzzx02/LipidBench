from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PeakTruthLab.scripts.annotation.recompute_rt_bounds_and_attrs import parse_args, recompute


if __name__ == "__main__":
    recompute(parse_args())
