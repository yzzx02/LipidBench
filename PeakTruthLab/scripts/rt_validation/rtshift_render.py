"""Process worker using the project's unchanged EIC plotting function."""
from pathlib import Path
import json
import runtime
import matplotlib
from lipidbench.utils.plot_eic import plot_eic

def render(task):
    rt,y,prior,out_dir,name=task
    out_dir=Path(out_dir)
    # Historical recovery_helpers imports the annotation module, which sets
    # these rcParams. Make the same rendering explicit in every spawn worker
    # so its PNG does not depend on import order or the entry-point script.
    with matplotlib.rc_context({'font.family':'sans-serif',
                                'font.sans-serif':['Arial','Microsoft YaHei','DejaVu Sans'],
                                'axes.unicode_minus':False}):
        plot_eic(rt,y,name,out_dir,xlim=(prior-1,prior+1),width_px=480,height_px=480,dpi=150,
                 normalize_y=False,rtmin=prior-.01,rtmax=prior+.01)
    jp=out_dir/f'{name}.json'
    label=json.loads(jp.read_text(encoding='utf8')); jp.unlink()
    x1=label['shapes'][0]['points'][0][0]; x2=label['shapes'][0]['points'][1][0]
    slope=.022/(x2-x1); intercept=prior-.011-slope*x1
    return str(out_dir/f'{name}.png'),slope,intercept
