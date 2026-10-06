"""Post-prediction diagnosis; never changes candidate selection or primary metrics."""
from pathlib import Path
import numpy as np
import pandas as pd

from runtime import PROJECT_ROOT as ROOT, WORK as W, OUTPUT as O


def selected_native_candidate(candidates, row):
    sub = candidates[candidates.feature_id.eq(row.feature_id)]
    match = (
        np.isclose(sub.pred_apex_rt, row.pred_apex_rt, rtol=0, atol=1e-9)
        & np.isclose(sub.pred_left, row.pred_left, rtol=0, atol=1e-9)
        & np.isclose(sub.pred_right, row.pred_right, rtol=0, atol=1e-9)
        & np.isclose(sub.pred_area, row.pred_area, rtol=1e-9, atol=1e-7)
    )
    sub = sub[match].sort_values('pred_score', ascending=False)
    assert len(sub), (row.feature_id, row.sample, row.rt_shift)
    return sub.iloc[0]


def main():
    d = pd.read_csv(O / 'per_feature_file_shift_results.csv')
    # A selected correct peak necessarily exists among the detector candidates.
    # Failed rows were independently checked against all saved candidates.
    d['target_candidate_present'] = (
        d.correct_detected.eq(1) | d.correct_candidate_in_output.fillna(0).eq(1)
    ).astype(int)
    d['selection_failure'] = d.target_candidate_present.eq(1) & d.correct_detected.eq(0)
    d['target_missing_from_valid_candidates'] = d.target_candidate_present.eq(0)

    def summarize(g):
        return pd.Series(dict(
            n_targets=len(g),
            n_target_candidates=int(g.target_candidate_present.sum()),
            target_candidate_recall=float(g.target_candidate_present.mean()),
            n_selected_correct=int(g.correct_detected.sum()),
            selected_peak_recall=float(g.correct_detected.mean()),
            n_selection_failures=int(g.selection_failure.sum()),
            n_targets_missing_from_valid_candidates=int(g.target_missing_from_valid_candidates.sum()),
            accurate_quantification_rate=float(g.accurate_quantification.mean()),
        ))

    for keys, filename in [
        (['method', 'rt_shift'], 'pipeline_stage_diagnostics.csv'),
        (['method', 'rt_shift', 'sample'], 'pipeline_stage_diagnostics_by_file.csv'),
    ]:
        result = d.groupby(keys).apply(summarize, include_groups=False).reset_index()
        for col in result.columns:
            if col.startswith('n_'):
                result[col] = result[col].astype(int)
        result.to_csv(O / filename, index=False)

    # Distinguish changed peak identity from boundary movement for native area failures.
    # Native peak table ordering was independently verified under every translation.
    native_failures = d[d.method.eq('xcms') & d.failure_category.eq('integration_area_error')]
    rows = []
    for row in native_failures.itertuples():
        shifted_candidates = pd.read_csv(
            W / f'predictions/xcms/shift_{row.rt_shift:+.1f}/{row.sample}_candidates.csv'
        )
        baseline_candidates = pd.read_csv(W / f'baseline_xcms/{row.sample}_candidates.csv')
        baseline = d[
            d.method.eq('xcms') & d.rt_shift.eq(0)
            & d.feature_id.eq(row.feature_id) & d['sample'].eq(row.sample)
        ].iloc[0]
        selected = selected_native_candidate(shifted_candidates, row)
        original = selected_native_candidate(baseline_candidates, baseline)
        rows.append(dict(
            feature_id=row.feature_id, sample=row.sample, rt_shift=row.rt_shift,
            area_error_pct=row.area_error_pct,
            baseline_native_index=int(original.candidate_index),
            selected_native_index=int(selected.candidate_index),
            native_peak_identity_changed=bool(selected.candidate_index != original.candidate_index),
            baseline_native_mz=original.pred_mz, selected_native_mz=selected.pred_mz,
            mass_difference_ppm=float((selected.pred_mz / original.pred_mz - 1) * 1e6),
            left_boundary_change_min=row.left_boundary_change_min,
            right_boundary_change_min=row.right_boundary_change_min,
            explanation='different native peak selected at nearly the same RT' if selected.candidate_index != original.candidate_index
                        else 'same native peak; inspect integration values',
        ))
    pd.DataFrame(rows).to_csv(O / 'xcms_area_failure_identity_diagnostics.csv', index=False)
    print(pd.read_csv(O / 'pipeline_stage_diagnostics.csv').to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
