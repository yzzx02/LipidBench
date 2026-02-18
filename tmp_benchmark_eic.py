from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile, PeakFileOptions

from lipidbench.eic.extract_eic_pyopenms import (
    extract_eic,
)


MZML = Path(r"D:\LipidBench\data\raw_mzML\SERUM-1_copy_P1-E4_1_step1_1.mzML")
N_FEATURES = 100
PPM = 10.0


if not MZML.exists():
    raise FileNotFoundError(MZML)

# Prepare feature table from first MS1 spectrum
exp = MSExperiment()
opts = PeakFileOptions()
opts.setMSLevels([1])
m = MzMLFile()
m.setOptions(opts)
m.load(str(MZML), exp)

first_ms1 = None
for s in exp:
    if int(s.getMSLevel()) == 1:
        mz_tmp, _ = s.get_peaks()
        if len(mz_tmp) == 0:
            continue
        first_ms1 = s
        break
if first_ms1 is None:
    raise RuntimeError("No MS1 spectrum found")

mzs, _ints = first_ms1.get_peaks()
mzs = np.asarray(mzs, dtype=np.float64)

# spread picks across spectrum to avoid duplicated m/z
idx = np.linspace(0, max(0, len(mzs) - 1), num=min(N_FEATURES, len(mzs)), dtype=int)
feature_mz = mzs[idx]
df_info = pd.DataFrame({"Feature_ID": [f"F{i+1}" for i in range(len(feature_mz))], "mz": feature_mz})


def run_nearest():
    return extract_eic(str(MZML), df_info, tolerance=PPM, unit="ppm", method="nearest")


def run_window_sum():
    return extract_eic(str(MZML), df_info, tolerance=PPM, unit="ppm", method="window_sum")


def bench(fn, repeat=1):
    times = []
    peaks = []
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        mat = fn()
        dt = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(dt)
        peaks.append(peak / (1024 * 1024))
    return mat, float(np.mean(times)), float(np.mean(peaks))


old_mat, old_t, old_mem = bench(run_nearest)
new_mat, new_t, new_mem = bench(run_window_sum)

same_shape = old_mat.shape == new_mat.shape
# nearest mode; should be numerically close
max_abs_diff = float(np.max(np.abs(old_mat - new_mat))) if same_shape else float("nan")

print("=== Benchmark (nearest vs window_sum, ppm) ===")
print(f"features: {len(df_info)}, scans: {old_mat.shape[1]}")
print(f"nearest     time={old_t:.4f}s  tracemalloc_peak={old_mem:.2f} MB")
print(f"window_sum  time={new_t:.4f}s  tracemalloc_peak={new_mem:.2f} MB")
print(f"same_shape={same_shape}, max_abs_diff={max_abs_diff:.6g}")
if new_t > 0:
    print(f"speedup(nearest/window_sum)={old_t/new_t:.2f}x")
