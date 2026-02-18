from bisect import bisect_left, bisect_right
from scipy.ndimage import gaussian_filter
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Union
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


def plot_eic(rt, intensity, name: str, folder_path: Union[str, Path]):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    fig: Figure = Figure(figsize=(4, 3))
    canvas: FigureCanvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(rt, intensity, linewidth=1.0, color="black")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='y', labelrotation=90)

    fig.tight_layout()
    out_path = folder / f"{name}.jpeg"
    canvas.print_jpeg(str(out_path))
    plt.close()