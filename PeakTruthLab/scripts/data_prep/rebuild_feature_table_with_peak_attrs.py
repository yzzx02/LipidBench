from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "rebuild_feature_table_with_peak_attrs.py"
    runpy.run_path(str(target), run_name="__main__")
