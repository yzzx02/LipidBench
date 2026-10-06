"""Evaluate completed predictions using the frozen baseline, with failure taxonomy."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

from runtime import PROJECT_ROOT as ROOT, WORK as W, OUTPUT as O
SHIFTS=[-.5,-.2,-.1,0.,.1,.2,.5];TOL=10/60
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(l,r,a,b):
    return np.maximum(0,np.minimum(r,b)-np.maximum(l,a))/(np.maximum(r,b)-np.minimum(l,a))

def summarize(d):
    e=d.area_error_pct.dropna(); c=d[d.correct_detected.eq(1)].area_error_pct.dropna()
    return pd.Series(dict(n_targets=len(d),n_found=int(d.pred_found.sum()),n_correct=int(d.correct_detected.sum()),
       recall=float(d.correct_detected.mean()),n_accurate_quantification=int(d.accurate_quantification.sum()),
       accurate_quantification_rate=float(d.accurate_quantification.mean()),n_area_errors=len(e),
       area_error_median_pct=float(e.median()),area_error_mean_pct=float(e.mean()),
       area_error_q1_pct=float(e.quantile(.25)),area_error_q3_pct=float(e.quantile(.75)),
       area_error_iqr_pct=float(e.quantile(.75)-e.quantile(.25)),
       correct_only_area_error_median_pct=float(c.median()),correct_only_area_error_mean_pct=float(c.mean())))

def main():
    freeze=json.loads((W/'freeze.json').read_text())
    assert digest(W/'reference_feature_table_frozen.csv')==freeze['reference_feature_table_sha256']
    assert digest(W/'baseline_references_frozen.csv')==freeze['baseline_references_sha256']
    table=pd.read_csv(W/'reference_feature_table_frozen.csv');ids=set(table.feature_id)
    refs=pd.read_csv(W/'baseline_references_frozen.csv');samples=pd.read_csv(W/'samples16.csv')['sample'].tolist()
    all_rows=[];candidate_paths={}
    for method in ['xcms','model']:
        for shift in SHIFTS:
            directory=W/(f'baseline_{method}' if shift==0 else f'predictions/{method}/shift_{shift:+.1f}')
            for sample in samples:
                path=directory/f'{sample}.csv'
                d=pd.read_csv(path);d=d[d.feature_id.isin(ids)].copy()
                assert len(d)==len(ids) and not d.feature_id.duplicated().any(),(method,shift,sample)
                d['method']=method;d['rt_shift']=shift
                candidate_paths[(method,shift,sample)]=directory/f'{sample}_candidates.csv'
                all_rows.append(d)
    pred=pd.concat(all_rows,ignore_index=True)
    assert len(pred)==len(ids)*16*7*2
    d=pred.merge(refs,on=['feature_id','sample','method'],validate='many_to_one').merge(table,on='feature_id',validate='many_to_one')
    d['expected_apex_rt']=d.common_baseline_apex_rt+d.rt_shift
    d['expected_left']=d.baseline_left+d.rt_shift;d['expected_right']=d.baseline_right+d.rt_shift
    d['apex_error_min']=d.pred_apex_rt-d.expected_apex_rt
    d['contains_expected_apex']=d.pred_left.le(d.expected_apex_rt)&d.pred_right.ge(d.expected_apex_rt)
    d['correct_detected']=(d.pred_found.eq(1)&d.apex_error_min.abs().le(TOL)&d.contains_expected_apex).astype(int)
    d['area_ratio']=d.pred_area/d.baseline_area
    d['signed_area_error_pct']=(d.area_ratio-1)*100
    d['area_error_pct']=d.signed_area_error_pct.abs()
    d.loc[d.pred_found.eq(0),['area_ratio','signed_area_error_pct','area_error_pct']]=np.nan
    d['accurate_quantification']=(d.correct_detected.eq(1)&d.area_error_pct.lt(20)).astype(int)
    d['own_baseline_shifted_boundary_iou']=overlap(d.pred_left,d.pred_right,d.expected_left,d.expected_right)
    d['left_boundary_change_min']=d.pred_left-d.expected_left;d['right_boundary_change_min']=d.pred_right-d.expected_right
    d['failure_category']='success'
    d.loc[d.pred_found.eq(0),'failure_category']='no_candidate'
    d.loc[d.pred_found.eq(1)&d.correct_detected.eq(0),'failure_category']='target_not_detected_or_selected'
    d.loc[d.correct_detected.eq(1)&d.area_error_pct.ge(20),'failure_category']='integration_area_error'
    d['correct_candidate_in_output']=np.nan
    failure=d.accurate_quantification.eq(0)
    for key,idx in d[failure].groupby(['method','rt_shift','sample']).groups.items():
        p=candidate_paths[key]
        try:cs=pd.read_csv(p)
        except (FileNotFoundError,pd.errors.EmptyDataError):cs=pd.DataFrame()
        for j in idx:
            row=d.loc[j];sub=cs[cs.feature_id.eq(row.feature_id)] if len(cs) else pd.DataFrame()
            if len(sub):
                good=(sub.pred_apex_rt-row.expected_apex_rt).abs().le(TOL)&sub.pred_left.le(row.expected_apex_rt)&sub.pred_right.ge(row.expected_apex_rt)
                has=bool(good.any())
            else:has=False
            d.loc[j,'correct_candidate_in_output']=int(has)
            if row.correct_detected==0 and has:d.loc[j,'failure_category']='selection_wrong_peak'
            elif row.correct_detected==0 and row.pred_found==1:
                d.loc[j,'failure_category']='boundary_excludes_expected_apex' if abs(row.apex_error_min)<=TOL else 'target_peak_not_in_candidates'
    d=d.sort_values(['method','rt_shift','sample','feature_id'])
    d.to_csv(O/'per_feature_file_shift_results.csv',index=False)
    d[d.accurate_quantification.eq(0)].to_csv(O/'failure_cases.csv',index=False)
    overall=d.groupby(['method','rt_shift'],sort=True).apply(summarize,include_groups=False).reset_index()
    perfile=d.groupby(['method','rt_shift','sample'],sort=True).apply(summarize,include_groups=False).reset_index()
    for summary in [overall,perfile]:
        for col in ['n_targets','n_found','n_correct','n_accurate_quantification','n_area_errors']:summary[col]=summary[col].astype(int)
    overall.to_csv(O/'summary_by_method_shift.csv',index=False)
    perfile.to_csv(O/'summary_by_file_method_shift.csv',index=False)
    d.groupby(['method','rt_shift','failure_category']).size().rename('n').reset_index().to_csv(O/'failure_category_counts.csv',index=False)
    cv=[]
    for (method,shift,fid),g in d.groupby(['method','rt_shift','feature_id']):
        ratios=g.area_ratio.dropna();areas=g.pred_area.dropna();base=g.baseline_area
        cv.append(dict(method=method,rt_shift=shift,feature_id=fid,n_found=len(ratios),n_correct=g.correct_detected.sum(),
           normalized_area_ratio_cv_pct=float(ratios.std(ddof=1)/ratios.mean()*100) if len(ratios)>1 else np.nan,
           raw_area_cv_pct=float(areas.std(ddof=1)/areas.mean()*100) if len(areas)>1 else np.nan,
           baseline_raw_area_cv_pct=float(base.std(ddof=1)/base.mean()*100)))
    pd.DataFrame(cv).to_csv(O/'feature_area_cv_across16.csv',index=False)
    baseline=d[d.rt_shift.eq(0)]
    assert baseline.correct_detected.eq(1).all() and baseline.accurate_quantification.eq(1).all()
    audit=dict(n_features=len(ids),n_files=16,n_records=len(d),n_baseline_records=len(baseline),
       both_methods_baseline_recall_one=True,frozen_hashes_pass=True,reference_priors_unchanged=True,
       nonzero_results_complete=True,recall_apex_tolerance_sec=10,correct_requires_boundary_contains_expected_apex=True,
       area_error_summary_population='all found selected candidates, including wrongly selected peaks; missing candidates are NaN; correct-only summaries also saved',
       accurate_quantification_denominator='all true feature x file targets; strictly <20 percent',
       no_rt_correction=True,no_gap_filling=True,no_retraining_or_shift_specific_tuning=True)
    (O/'evaluation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(overall[['method','rt_shift','recall','area_error_median_pct','area_error_mean_pct','accurate_quantification_rate']].to_string(index=False),flush=True)

if __name__=='__main__':main()
