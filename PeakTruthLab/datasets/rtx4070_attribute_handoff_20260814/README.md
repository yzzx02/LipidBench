# RTX4070 old-final attribute handoff

This directory is a portable, frozen calculation manifest for the workstation holding the original mzML sources. It contains no raw spectra and authorizes no training or label changes.

## Contents

- `handoff_manifest.json`: scope, policies, input provenance, counts, and input hashes.
- `source_inventory.csv`: the 29 exact mzML basenames and expected job counts.
- `old_final_seed_attribute_jobs.csv`: 15,317 Seed calculation jobs.
- `old_final_peak_attribute_jobs.csv`: 18,580 final peak-instance calculation jobs.

Follow `PeakTruthLab/docs/RTX4070_REMOTE_ATTRIBUTE_HANDOFF.md`. The remote computation entry point is `PeakTruthLab/scripts/data_prep/compute_rtx4070_attribute_handoff.py`.

## Frozen interpretation

- Seed legacy 13 attributes are authoritative; calculate `SYM`, `MOD`, and `EDGE` only.
- Matched peak legacy 13 attributes are authoritative; calculate `SYM`, `MOD`, and `EDGE` on the final human box.
- For 54 new peak instances without a legacy row, calculate all 16 attributes.
- The peak list contains 18,575 v2-manifest boxes and five additional reviewed `OUT_FIG` LabelMe shapes, for 18,580 total jobs.
- Human labels and boundaries must never be refined or replaced by the calculation workstation.

CSV files use UTF-8 with a BOM. `job_id` is the immutable join key. `sample_id` identifies the chromatogram image/window, `source_mzml_name` is the required exact spectrum filename, and `old_split` records the frozen old-final split for provenance only.
