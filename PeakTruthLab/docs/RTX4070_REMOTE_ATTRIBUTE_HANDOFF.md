# RTX4070 workstation handoff: old-final attribute completion

## Scope and stop conditions

This handoff completes the attribute tables for the frozen old-final dataset only. It does not merge the later 4,500-sample package, alter a label or box boundary, change a split, read Test for model selection, or start training.

The workstation with the original mzML files must calculate the three missing attributes `SYM`, `MOD`, and `EDGE` for all old-final Seed candidates and final peak instances. For peak instances without a legacy row, it must calculate all 16 attributes. Existing legacy values and the final human annotations remain authoritative.

Stop without uploading a completed result if any required mzML source cannot be resolved exactly, duplicate filenames have different hashes, a job fails, an expected row is missing or duplicated, or `remote_attribute_qc.json` reports a status other than `ok`.

## Frozen inputs

Checkout branch `agent/rtx4070-final-merge-split` and use:

- `PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/handoff_manifest.json`
- `PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/source_inventory.csv`
- `PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/old_final_seed_attribute_jobs.csv`
- `PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/old_final_peak_attribute_jobs.csv`

Expected jobs:

- Seed candidates: 15,317
- Peak instances: 18,580
- True peaks: 18,487
- `OUT_FIG` peaks: 93
- Required mzML files: 29

The 18,580 peak jobs consist of 18,575 boxes in the frozen v2 manifests plus five reviewed `OUT_FIG` shapes that were present in the final LabelMe annotations but omitted from the detector manifests. They are retained intentionally.

Verify the three input-file SHA-256 values against `handoff_manifest.json` before computation. Do not edit or regenerate a job CSV on the remote workstation.

## Attribute policy

The fixed order is:

`SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`

- Seed rows: preserve the legacy 13 values and calculate only `SYM`, `MOD`, and `EDGE` for the final merge.
- Peaks with `legacy_match_status` equal to `exact` or `adjusted`: preserve the legacy 13 and calculate `SYM`, `MOD`, and `EDGE` using the final human box.
- Peaks with `legacy_match_status` equal to `new_no_legacy`: calculate all 16 attributes because no legacy peak-instance values exist.
- Never refine, replace, or infer a new human boundary.
- Preserve missing/non-finite values in output and report them in QC; do not impute them.

The computation script imports the repository implementation in `lipidbench/utils/peak_attributes.py`. Do not replace it with a new formula.

## Workstation procedure

1. Pull the repository and checkout `agent/rtx4070-final-merge-split`.
2. Read this file, the handoff manifest, and the source inventory. Verify branch, Git status, row counts, unique `job_id` counts, and SHA-256 values.
3. Locate the directory or directories containing the 29 original mzML files. The script searches recursively and matches exact basenames. If duplicates exist, it accepts them only when their SHA-256 hashes are identical. Do not substitute a similarly named file and do not reconvert raw data when the original converted mzML exists.
4. Use the repository environment, or create an isolated Python environment with `numpy`, `pandas`, `scipy`, and either `pyopenms` or `pymzml`. Confirm that `lipidbench/utils/peak_attributes.py` imports successfully.
5. Run the computation. Replace the example source and output paths with actual local absolute paths. Repeat `--source-root` when sources span multiple directories.

```powershell
python PeakTruthLab/scripts/data_prep/compute_rtx4070_attribute_handoff.py `
  --source-root "D:\path\to\mzML" `
  --output-dir "D:\CODE\rtx4070_oldfinal_attribute_results" `
  --backend auto
```

6. The script writes restartable per-source files under `parts`. If interrupted, rerun the same command. After completion, inspect all final artifacts:

   - `source_resolution.csv`
   - `old_final_seed_calculated_16.csv`
   - `old_final_peak_calculated_16.csv`
   - `remote_attribute_qc.json`
   - `SHA256SUMS.txt`

7. Completion is valid only when:

   - `remote_attribute_qc.json` has `status: "ok"`;
   - Seed output has 15,317 unique jobs;
   - peak output has 18,580 unique jobs;
   - all 29 mzML sources resolve;
   - no computation job has an error;
   - recalculated legacy 13 values pass the built-in regression comparison for every comparable Seed value and every exact legacy peak match.

8. Do not upload raw mzML files or the intermediate `parts` directory. Create branch `codex/rtx4070-oldfinal-16attr-results`, copy only the five final artifacts into `PeakTruthLab/results/rtx4070_oldfinal_attribute_results_20260814/`, force-add them if required by `.gitignore`, commit, push, and open a pull request with base branch `agent/rtx4070-final-merge-split`.
9. In the pull request body, report the exact command, Python/package versions, backend, source roots, elapsed time, counts, QC status, missing/non-finite counts, and output SHA-256 values. Do not describe the task as completed if QC is blocked.
10. Stop after uploading the result pull request. Do not merge datasets, split data, or train a model. Return the pull request URL to the user.

## What happens after upload

The current workstation will fetch the result pull request, independently verify hashes, counts, source resolution, job completeness, legacy regression checks, label/boundary preservation, and QC status. Only after that verification will the 16-attribute tables be merged with the later package and the frozen split workflow resume.
