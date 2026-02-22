from bisect import bisect_left, bisect_right
from scipy.ndimage import gaussian_filter
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Union, Optional
def calc_coordinate(df_info,intensities,rt,k,windows_size=2):
    feature_rt = df_info[k]['RT']
    left_rt = feature_rt - windows_size/2 if feature_rt - windows_size/2 > rt.min() else rt.min()
    right_rt = feature_rt + windows_size/2 if feature_rt + windows_size/2 < rt.max() else rt.max()
    left_idx = bisect_left(rt, left_rt)
    right_idx = bisect_right(rt, right_rt)
    if right_idx - left_idx >= 0:
        calc_intensity = intensities[left_idx:right_idx]
        calc_rt = rt[left_idx:right_idx]
    return calc_intensity, calc_rt

def gussian_smooth(intensity,rt,sigma):
    if sigma == 0:
        return intensity, rt
    else:
        return gaussian_filter(intensity, sigma=sigma), rt


def plot_eic(
    rt,
    intensity,
    name: str,
    folder_path: Union[str, Path],
    *,
    xlim: Optional[tuple[float, float]] = None,
    width_px: int = 400,
    height_px: int = 300,
    dpi: int = 100,
    normalize_y: bool = False,
):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    fig_w = max(1, int(width_px)) / float(dpi)
    fig_h = max(1, int(height_px)) / float(dpi)
    fig: Figure = Figure(figsize=(fig_w, fig_h), dpi=dpi)
    canvas: FigureCanvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    y = intensity
    if normalize_y:
        try:
            import numpy as np

            y_np = np.asarray(intensity, dtype=float)
            vmax = float(np.nanmax(y_np)) if y_np.size else 0.0
            y = (y_np / vmax) if vmax > 0 else y_np
        except Exception:
            y = intensity

    ax.plot(rt, y, linewidth=1.0, color="black")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if normalize_y:
        ax.set_ylim(0.0, 1.0)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='y', labelrotation=90)

    fig.tight_layout()
    out_path = folder / f"{name}.jpeg"
    canvas.print_jpeg(str(out_path))
    plt.close()