"""Fixed-prior inference on actual original or RT-shifted mzML, using existing detector helpers."""
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from recovery_helpers import load_model,load_ms1_spectra,_extract_trace,valid_candidates,sha256
from rtshift_render import render
from rtshift_trace import extract_batch

from runtime import PROJECT_ROOT as ROOT, WORK, CHECKPOINT, resolve_device

def run_group(model,device,pool,sample,path,table,shift,out,batch_size):
    start=time.perf_counter()
    spectra=load_ms1_spectra(path,backend='pyopenms')
    tasks=[]; items=[]; rows=[]; all_candidates=[]
    image_dir=out/'eic_images'/sample
    image_dir.mkdir(parents=True,exist_ok=True)
    # Same nearest-centroid rule, batched over targets, checked against the legacy extractor.
    batch_rt,batch_y=extract_batch(spectra,table.reference_mz.to_numpy())
    for row_index,row in enumerate(table.itertuples(index=False)):
        full_rt,full_y=batch_rt,batch_y[row_index]
        prior=float(row.reference_rt)
        mask=(full_rt>=prior-1)&(full_rt<=prior+1)
        rt=full_rt[mask]; y=full_y[mask]
        name=f'{row.feature_id}_{shift:+.1f}'
        tasks.append((rt,y,prior,str(image_dir),name))
        items.append((row.feature_id,prior,rt,y,full_rt,full_y))
    del spectra
    rendered=pool.map(render,tasks,chunksize=8)
    pending=[]
    def flush():
        if not pending:return
        tensors=[to_tensor(Image.open(v[1][0]).convert('RGB')).to(device) for v in pending]
        with torch.inference_mode(): detections=model.detector(tensors)
        for (item,im),detection in zip(pending,detections,strict=True):
            feature_id,prior,rt,y,fr,fy=item
            cs=valid_candidates(detection,rt,y,fr,fy,prior,im[1],im[2])
            for k,c in enumerate(cs):all_candidates.append(dict(feature_id=feature_id,sample=sample,rt_shift=shift,candidate_index=k,**c))
            cs.sort(key=lambda c:(round(abs(c['pred_apex_rt']-prior),3),-c['pred_score']))
            rec=dict(feature_id=feature_id,sample=sample,rt_shift=shift,pred_found=int(bool(cs)),valid_candidate_count=len(cs))
            rec.update(cs[0] if cs else dict(pred_score=np.nan,pred_apex_rt=np.nan,pred_left=np.nan,pred_right=np.nan,pred_area=np.nan))
            rows.append(rec)
        pending.clear()
    for index,(item,im) in enumerate(zip(items,rendered,strict=True),1):
        pending.append((item,im))
        if len(pending)>=batch_size:flush()
        if index%200==0: print(f'model {sample} shift={shift:+.1f}: {index}/{len(items)} ({time.perf_counter()-start:.1f}s)',flush=True)
    flush()
    result=pd.DataFrame(rows)
    result.to_csv(out/f'{sample}.csv',index=False)
    pd.DataFrame(all_candidates).to_csv(out/f'{sample}_candidates.csv',index=False)
    print(f'model completed {sample} shift={shift:+.1f}: {len(rows)}, {time.perf_counter()-start:.1f}s',flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--shifts',nargs='+',type=float)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--limit',type=int)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);p.add_argument('--table',type=Path);p.add_argument('--out-tag',default='predictions')
    a=p.parse_args()
    table_path=a.table or WORK/('candidate_feature_table.csv' if a.baseline else 'reference_feature_table_frozen.csv')
    table=pd.read_csv(table_path)
    if a.limit:table=table.iloc[:a.limit]
    if any(c.startswith('ref_area') or c.startswith('baseline_') for c in table.columns):raise ValueError('inference table contains outcome references')
    manifest=pd.read_csv(WORK/'shifted_mzml_manifest.csv')
    device=resolve_device(a.device);torch.set_num_threads(2)
    model,info=load_model(CHECKPOINT,device)
    shifts=[0.] if a.baseline else (a.shifts or [-.5,-.2,-.1,.1,.2,.5])
    samples=pd.read_csv(WORK/'samples16.csv')['sample'].tolist()
    active=table.copy()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for shift in shifts:
            out=WORK/('baseline_model' if a.baseline else f'{a.out_tag}/model/shift_{shift:+.1f}')
            out.mkdir(parents=True,exist_ok=True)
            for sample in samples:
                path=manifest[(manifest['sample']==sample)&(manifest.rt_shift==shift)].iloc[0].mzml_path
                fn=out/f'{sample}.csv'
                if fn.exists():
                    result=pd.read_csv(fn);print('reuse model',sample,shift,flush=True)
                else:result=run_group(model,device,pool,sample,path,active if a.baseline else table,shift,out,a.batch_size)
                if a.baseline:
                    # Safe early rejection: only baseline model existence. Full joint QC happens after both baselines.
                    good=set(result[result.pred_found==1].feature_id)
                    active=active[active.feature_id.isin(good)]
                    print('baseline model features remaining',len(active),flush=True)
            (out/'provenance.json').write_text(json.dumps(dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),
                model_info=info,shift=shift,n_input_features=len(table),eic_window_min=2,eic_method='nearest',eic_ppm=10,
                score_threshold=.5,candidate_rule='nearest predicted apex to unchanged reference RT; score tie break',
                input_feature_table_sha256=sha256(table_path)),indent=2),encoding='utf8')

if __name__=='__main__':main()
