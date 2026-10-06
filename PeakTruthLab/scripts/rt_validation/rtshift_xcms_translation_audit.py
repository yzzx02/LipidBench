"""Post-inference check of native full-spectrum centWave's response to a constant RT translation."""
from pathlib import Path
import numpy as np,pandas as pd
from runtime import PROJECT_ROOT as ROOT, WORK as W, OUTPUT as O

def main():
    rows=[]
    for sample in pd.read_csv(W/'samples16.csv')['sample']:
        base=pd.read_csv(W/'baseline_xcms'/f'{sample}_all_chromPeaks.csv')
        for shift in [-.5,-.2,-.1,.1,.2,.5]:
            other=pd.read_csv(W/f'predictions/xcms/shift_{shift:+.1f}'/f'{sample}_all_chromPeaks.csv')
            rec=dict(sample=sample,rt_shift=shift,n_native_peaks_baseline=len(base),n_native_peaks_shift=len(other),same_peak_count=len(base)==len(other))
            if len(base)==len(other) and np.allclose(base.mz,other.mz,rtol=1e-12,atol=1e-10):
                rt_error=np.max(np.abs(other[['rt','rtmin','rtmax']].to_numpy()-base[['rt','rtmin','rtmax']].to_numpy()-shift*60))
                error=np.abs(other.into.to_numpy()-base.into.to_numpy())/np.maximum(base.into.to_numpy(),1)
                rec.update(ordered_mz_match=True,max_native_rt_translation_error_sec=float(rt_error),
                   max_native_area_relative_error_pct=float(error.max()*100),
                   fraction_native_areas_agree_rel1e8=float((error<1e-8).mean()))
            else:rec['ordered_mz_match']=False
            rows.append(rec)
    pd.DataFrame(rows).to_csv(O/'xcms_native_translation_audit.csv',index=False)
    print(pd.DataFrame(rows).describe(include='all').to_string(),flush=True)

if __name__=='__main__':main()
