from bisect import bisect_left, bisect_right
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, ScalarFormatter, MaxNLocator
from pathlib import Path
from typing import Union, Optional


def _tick_formatter(v, _pos):
    if abs(v) < 1e-12:
        v = 0.0
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def _nice_step(target_step: float) -> float:
    if target_step <= 0:
        return 0.5
    exp = np.floor(np.log10(target_step))
    base = target_step / (10 ** exp)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 2.5:
        nice = 2.5
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * (10 ** exp))
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


def generate_labelme_json(json_path, image_name, rtmin, rtmax, rt_arr, eic_arr, ax, fig, use_tight: bool = False):
    rt_arr = np.asarray(rt_arr, dtype=float)
    eic_arr = np.asarray(eic_arr, dtype=float)

    if rt_arr.size == 0 or eic_arr.size == 0:
        return

    lo = float(min(rtmin, rtmax))
    hi = float(max(rtmin, rtmax))

    # X-axis padding: expand both sides by 5% of box width.
    width = hi - lo
    pad = width * 0.05 if width > 0 else 0.0
    rt_start_final = lo - pad
    rt_end_final = hi + pad

    mask = (rt_arr >= rt_start_final) & (rt_arr <= rt_end_final)
    if np.any(mask):
        intensity_max = float(np.max(eic_arr[mask])) * 1.05
    else:
        intensity_max = float(np.max(eic_arr)) * 1.05

    # Y-axis rule:
    # - bottom is anchored to physical baseline 0
    # - top follows local max intensity in interval
    intensity_min = 0.0
    intensity_max = max(intensity_max, 0.0)

    # data -> display (in full figure canvas pixel coordinates)
    p1_disp = ax.transData.transform((rt_start_final, intensity_max))
    p2_disp = ax.transData.transform((rt_end_final, intensity_min))

    if use_tight:
        # Map display coords into cropped image pixel coordinates.
        dpi = float(fig.dpi)
        renderer = fig.canvas.get_renderer()
        tight_bbox_in = fig.get_tightbbox(renderer).padded(0.1)  # inches
        x0_px = float(tight_bbox_in.x0 * dpi)
        y0_px = float(tight_bbox_in.y0 * dpi)
        width = max(1.0, float(tight_bbox_in.width * dpi))
        height = max(1.0, float(tight_bbox_in.height * dpi))

        x1_disp_cropped = float(p1_disp[0] - x0_px)
        y1_disp_cropped = float(p1_disp[1] - y0_px)
        x2_disp_cropped = float(p2_disp[0] - x0_px)
        y2_disp_cropped = float(p2_disp[1] - y0_px)
    else:
        # Fixed-size canvas (no tight crop).
        width, height = fig.canvas.get_width_height()
        width = max(1.0, float(width))
        height = max(1.0, float(height))
        x1_disp_cropped = float(p1_disp[0])
        y1_disp_cropped = float(p1_disp[1])
        x2_disp_cropped = float(p2_disp[0])
        y2_disp_cropped = float(p2_disp[1])

    # display(bottom-left origin) -> labelme(top-left origin)
    x1, y1 = x1_disp_cropped, float(height - y1_disp_cropped)
    x2, y2 = x2_disp_cropped, float(height - y2_disp_cropped)

    # normalize rectangle corners and clip to image boundary
    left = max(0.0, min(float(width), min(x1, x2)))
    right = max(0.0, min(float(width), max(x1, x2)))
    top = max(0.0, min(float(height), min(y1, y2)))
    bottom = max(0.0, min(float(height), max(y1, y2)))

    labelme = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [
            {
                "label": "True_Peak",
                "points": [[left, top], [right, bottom]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        ],
        "imagePath": str(image_name),
        "imageData": None,
        "imageHeight": int(round(height)),
        "imageWidth": int(round(width)),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme, f, ensure_ascii=False, indent=2)


def plot_eic(
    rt,
    intensity,
    name: str,
    folder_path: Union[str, Path],
    *,
    xlim: Optional[tuple[float, float]] = None,
    width_px: int = 400,
    height_px: int = 300,
    dpi: int = 150,
    normalize_y: bool = False,
    rtmin: Optional[float] = None,
    rtmax: Optional[float] = None,
):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    # Standardized lightweight canvas
    fig = plt.figure(figsize=(float(width_px) / float(dpi), float(height_px) / float(dpi)), dpi=int(dpi))
    ax = fig.add_subplot(111)
    y = intensity
    if normalize_y:
        try:
            y_np = np.asarray(intensity, dtype=float)
            vmax = float(np.nanmax(y_np)) if y_np.size else 0.0
            y = (y_np / vmax) if vmax > 0 else y_np
        except Exception:
            y = intensity

    ax.plot(rt, y, color="royalblue", linewidth=1.0)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if normalize_y:
        y_top = 1.0
        y_bottom = -0.08
        ax.set_ylim(y_bottom, y_top)
        y_step = 0.5
    else:
        y_arr = np.asarray(y, dtype=float)
        y_max = float(np.max(y_arr)) if y_arr.size else 0.0
        y_top = max(y_max * 1.15, 0.5)
        y_bottom = -0.08 * y_top
        ax.set_ylim(y_bottom, y_top)
        y_range = max(y_top - y_bottom, 0.5)
        y_step = max(0.5, _nice_step(y_range / 5.0))

    # Fixed tick spacing for consistent 2-min window visualization.
    x_step = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(_tick_formatter))
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='both', labelsize=6)
    ax.yaxis.get_offset_text().set_size(6)

    fig.tight_layout()
    fig.canvas.draw()
    out_path = folder / f"{name}.jpeg"
    plt.savefig(str(out_path), dpi=int(dpi))

    if rtmin is not None and rtmax is not None:
        json_path = folder / f"{name}.json"
        generate_labelme_json(
            json_path=json_path,
            image_name=out_path.name,
            rtmin=float(rtmin),
            rtmax=float(rtmax),
            rt_arr=rt,
            eic_arr=y,
            ax=ax,
            fig=fig,
            use_tight=False,
        )

    plt.close()