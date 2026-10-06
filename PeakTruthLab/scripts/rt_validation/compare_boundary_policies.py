"""Replay the exploratory boundary audit from saved raw EICs, without a GPU."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import runtime
from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries, refine_peak_boundaries_guarded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    inputs = pd.read_csv(args.audit_dir / 'paired_inputs_and_manual_references.csv', usecols=[
        'image_id', 'domain_id', 'xcms_raw_apex_rt_min', 'xcms_raw_left_rt_min', 'xcms_raw_right_rt_min'])
    predictions = []
    for row in inputs.itertuples():
        with np.load(args.audit_dir / 'eic_cache' / f'{row.image_id}.npz') as trace:
            rt, eic = trace['rt'], trace['eic']
        hints = dict(rtmin_hint=row.xcms_raw_left_rt_min, rtmax_hint=row.xcms_raw_right_rt_min)
        core = refine_peak_boundaries(rt, eic, row.xcms_raw_apex_rt_min, **hints)
        guard = refine_peak_boundaries_guarded(rt, eic, row.xcms_raw_apex_rt_min, **hints)
        for name, left, right, apex, status in [
            ('native_xcms', row.xcms_raw_left_rt_min, row.xcms_raw_right_rt_min, row.xcms_raw_apex_rt_min, 'ok'),
            ('main_refiner', core.rtmin, core.rtmax, core.apex_rt, core.status),
            ('main_refiner_one_scan_guard', guard.rtmin, guard.rtmax, guard.apex_rt, guard.status),
        ]:
            predictions.append(dict(image_id=row.image_id, domain_id=row.domain_id, variant=name,
                                    pred_left=left, pred_right=right, pred_apex=apex, status=status))
    # All new boundary predictions are complete before accessing manual bounds.
    truth = pd.read_csv(args.audit_dir / 'paired_inputs_and_manual_references.csv', usecols=[
        'image_id', 'peak_left_rt_min', 'peak_right_rt_min', 'model_left_rt_min', 'model_right_rt_min'])
    d = pd.DataFrame(predictions).merge(truth, on='image_id', validate='many_to_one')
    model = inputs[['image_id', 'domain_id']].merge(truth, on='image_id', validate='one_to_one')
    model = model.assign(variant='locked_model', pred_left=model.model_left_rt_min,
                         pred_right=model.model_right_rt_min, pred_apex=np.nan, status='archived_prediction')
    d = pd.concat([d, model], ignore_index=True)
    d['left_abs_error_min'] = (d.pred_left - d.peak_left_rt_min).abs()
    d['right_abs_error_min'] = (d.pred_right - d.peak_right_rt_min).abs()
    intersection = np.maximum(0, np.minimum(d.pred_right, d.peak_right_rt_min) - np.maximum(d.pred_left, d.peak_left_rt_min))
    union = np.maximum(d.pred_right, d.peak_right_rt_min) - np.minimum(d.pred_left, d.peak_left_rt_min)
    d['boundary_iou'] = intersection / union
    d['width_ratio'] = (d.pred_right - d.pred_left) / (d.peak_right_rt_min - d.peak_left_rt_min)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.output_dir / 'paired_boundary_policy_results.csv', index=False)
    summaries = []
    for group_keys in [['variant'], ['domain_id', 'variant']]:
        for key, g in d.groupby(group_keys):
            key = key if isinstance(key, tuple) else (key,)
            errors = np.r_[g.left_abs_error_min, g.right_abs_error_min]
            summaries.append(dict(zip(group_keys, key)) | dict(
                domain_id=dict(zip(group_keys, key)).get('domain_id', 'ALL'), n_pairs=len(g),
                median_boundary_side_error_min=float(np.median(errors)), mean_boundary_side_error_min=float(errors.mean()),
                median_boundary_iou=float(g.boundary_iou.median()), fraction_iou_ge_0p5=float(g.boundary_iou.ge(.5).mean()),
                median_width_ratio=float(g.width_ratio.median()), n_guard_qc_fail=int(g.status.str.startswith('guard_').sum())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / 'boundary_policy_summary.csv', index=False)
    print(summary[summary.domain_id.eq('ALL')].to_string(index=False))


if __name__ == '__main__':
    main()
