# PeakTruthLab final merged dataset (2026-08-14)

This is a new, portable merged directory. The protected old-final dataset was not overwritten.

- Seed master: `tables/seed_master_16attrs.csv`
- Peak-instance master: `tables/peak_instances_master_16attrs.csv`
- Locked main split: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- Test lock: `splits/test_manifest_lock.json`
- Audits and QC: `audits/`
- Images and normalized LabelMe JSON: `images/`

All 16 attributes are raw values. Missing old-final states are preserved. Imputation and standardization must be fitted on Train only.
