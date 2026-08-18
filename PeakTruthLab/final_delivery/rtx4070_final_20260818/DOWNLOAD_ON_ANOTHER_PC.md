# Download on the mzML workstation

Open the GitHub Release `rtx4070-final-20260818` and download the assets you need:

- model inference: `LipidBench_Main_model_weights_20260818.zip`
- raw experiment review: `LipidBench_second_experiment_results_no_fold_weights_20260818.zip`
- images, LabelMe annotations, raw attributes, splits, and audits: `PeakTruthLab_final_dataset_20260814.zip`

Verify every asset against `release_manifest.csv` or `release_manifest.json` before extracting. On PowerShell:

```powershell
Get-FileHash .\LipidBench_Main_model_weights_20260818.zip -Algorithm SHA256
Get-FileHash .\LipidBench_second_experiment_results_no_fold_weights_20260818.zip -Algorithm SHA256
Get-FileHash .\PeakTruthLab_final_dataset_20260814.zip -Algorithm SHA256
```

After cloning the repository, extract the dataset ZIP so that its directory is available as `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814/`.

The stored image and annotation paths are relative, so no `D:` drive rewriting is required. Keep local mzML files outside Git and use the `source_mzML_name` and `source_mzML_sha256` fields to match them to dataset rows.
