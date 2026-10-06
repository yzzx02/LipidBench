from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,subprocess,time
import pandas as pd
from runtime import PROJECT_ROOT as ROOT, WORK, RSCRIPT, SCRIPT_DIR

def job(sample,path,shift,table,out):
    out.mkdir(parents=True,exist_ok=True);fn=out/f'{sample}.csv'
    if fn.exists():return f'reuse XCMS {sample} {shift:+.1f}'
    start=time.perf_counter()
    p=subprocess.run([str(RSCRIPT),str(SCRIPT_DIR/'rtshift_xcms_full.R'),path,str(table),str(fn),sample],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/f'{sample}.log').write_text(p.stdout,encoding='utf8')
    if p.returncode:raise RuntimeError(f'{sample} {shift}: {p.stdout[-3000:]}')
    return f'XCMS completed {sample} {shift:+.1f}: {time.perf_counter()-start:.1f}s'

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    table=WORK/('candidate_feature_table.csv' if a.baseline else 'reference_feature_table_frozen.csv')
    man=pd.read_csv(WORK/'shifted_mzml_manifest.csv');man=man[man.rt_shift.eq(0) if a.baseline else man.rt_shift.ne(0)]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs=[]
        for r in man.itertuples(index=False):
            out=WORK/('baseline_xcms' if a.baseline else f'predictions/xcms/shift_{r.rt_shift:+.1f}')
            fs.append(pool.submit(job,r.sample,r.mzml_path,r.rt_shift,table,out))
        for f in as_completed(fs):print(f.result(),flush=True)

if __name__=='__main__':main()
