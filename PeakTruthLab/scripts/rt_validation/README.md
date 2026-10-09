# ChromaPeak RT validation: transfer and replay

This workflow replays the **2026-09-30 actual-RT experiment** on the post-PR #9
ChromaPeak code. It keeps the same 16 original mzML files, 1,000 jointly detected
baseline features, fixed reference priors, historical image-only checkpoint,
score threshold and seven signed RT shifts. It does not reselect targets,
retrain, adjust shift-specific parameters or apply RT alignment.

## Download and extract the Release assets

[RT validation Release](https://github.com/yzzx02/LipidBench/releases/tag/chromapeak-rt-validation-20260930)

- `ChromaPeak_RT_complete_results_20260930.zip`: all result CSVs, frozen
  references, raw candidates/logs, historical scripts and exploratory boundary
  audit with cached raw EICs. Figures are omitted.
- `ChromaPeak_RT_mzml16_inputs.zip`: the exact 16 **original, unshifted** mzML
  files and their SHA-256 manifest. Shifted files are regenerated locally.
- `ChromaPeak_RT_image_only_checkpoint_20260907.zip`: the exact checkpoint
  used in this experiment. Do not substitute the project's current concat
  checkpoint; its SHA-256 is checked before prediction.
- `ChromaPeak_RT_R461_xcms4101_windows.zip`: the archived Windows R 4.6.1
  runtime including XCMS 4.10.1, MSnbase, mzR and package documentation/licenses.
- `release_assets_sha256.json`: archive hashes and CRC verification records.

Use ASCII paths such as `D:\ChromaPeakData`. Keep archival files intact. Historical
README files record the old workstation paths; use the commands below to replay.
Keep Arial available for the historical EIC font. Rendering now sets the original
font policy explicitly in every worker; the preflight records the resolved font
and software/source fingerprint. A change of device or environment cannot silently
reuse existing prediction CSVs in the same work directory.

On current main, paper deliverables and their sample manifests are machine-local,
not source files. `init` accepts `--sample-manifest <path-to-mzml16_manifest.csv>`;
without it, the default is
`<CHROMAPEAK_LOCAL_ROOT>/artifacts/reproduction/PeakTruthLab/final_delivery/rt_validation_20260930/mzml16_manifest.csv`.
The historical Release tag retains its original manifest. Runtime work and caches
should use an external `--work-dir`; the fallback runtime location is external too.

## RX 9070 XT on Windows

Use Python **3.12** and a matching AMD **torch + torchvision** pair. The checked
AMD ROCm 7.2.1 instructions list RX 9070 XT / Windows 11 and torch 2.9.1 with
torchvision 0.24.1. Install the ROCm runtime wheels and framework wheels from
[AMD's installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html),
including the indicated graphics driver. Compatibility source:
[AMD Windows matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html).
These are a verified documented configuration, not a claim that the entire
ROCm stack or this experiment has already been tested on the user's AMD machine.

Then install project dependencies without replacing the AMD framework wheels:

```powershell
python -m pip install -r requirements-rt-validation.txt
```

ROCm uses `torch.cuda` and `--device cuda` in this code. The preflight explicitly
checks GPU NMS, RoIAlign, strict checkpoint loading and an actual original EIC.
If native detection operators fail, fix the torch/torchvision installation
before the full run. `--device cpu` is an explicit slower diagnostic alternative.
No NVIDIA-only dependency file or DirectML conversion is used here.

## Initialize and run

Clone the Release tag to obtain this complete source version:

```powershell
git clone --branch chromapeak-rt-validation-20260930 https://github.com/yzzx02/LipidBench.git ChromaPeak
cd ChromaPeak
```

Example extracted directories:

```text
D:\ChromaPeakData\results\rt_shift16\...
D:\ChromaPeakData\mzml16\*.mzML
D:\ChromaPeakData\checkpoint\best_detection.pt
D:\ChromaPeakData\R-4.6.1-xcms4.10.1\bin\Rscript.exe
```

```powershell
python PeakTruthLab/scripts/rt_validation/run.py init --work-dir D:\ChromaPeakData\replay9070 --bundle D:\ChromaPeakData\results --mzml-dir D:\ChromaPeakData\mzml16 --checkpoint D:\ChromaPeakData\checkpoint\best_detection.pt --rscript D:\ChromaPeakData\R-4.6.1-xcms4.10.1\bin\Rscript.exe
python PeakTruthLab/scripts/rt_validation/run.py preflight --work-dir D:\ChromaPeakData\replay9070 --device cuda
python PeakTruthLab/scripts/rt_validation/run.py all --work-dir D:\ChromaPeakData\replay9070 --device cuda --workers 2 --batch-size 4
```

`all`: preflight → regenerate/audit 112 actual mzML files → both baselines →
baseline compatibility gate → both nonzero predictions → evaluation → native
XCMS invariance audit → failure-stage diagnostics → comparison with RTX4070.
Each stage can also be run separately. An initialized directory cannot be
overwritten by `init`; use another directory for a new run.

The baseline gate compares all 32,000 records against the frozen identities and
areas. If identity or area error ≥20% occurs, stop and inspect
`results/baseline_hardware_comparison.csv`. This prevents a broken baseline from
being interpreted as an RT-shift effect. For an exact replay XCMS must be 4.10.1.
Cached prediction CSVs support interrupted-run resume within this work directory;
do not mix checkpoints, framework versions or different devices in one run.

Main outputs are under the selected work directory's `results/`:

- `hardware_preflight.json` and `baseline_hardware_comparison.csv`;
- `per_feature_file_shift_results.csv` (224,000 rows);
- `summary_by_method_shift.csv`, `summary_by_file_method_shift.csv`;
- `failure_cases.csv`, `pipeline_stage_diagnostics.csv`;
- `hardware_summary_comparison.csv` with signed percentage-point changes;
- mzML and native XCMS integrity audits.

Areas in the replay are normalized to the original frozen own-method baseline
areas. Baseline replay differences are separately reported; reference values
are not silently updated to make a new GPU agree. Exact floating-point agreement
between devices is not assumed.

## Boundary refinement policy

PR #9 retains the original algorithm and fixes short-trace smoothing, invalid
RT axes and nonfinite hints. Its normal-trace bounds match the pre-PR core on
all 473 earlier paired peaks. The historical recovery experiment additionally
capped outward extension at one median MS1 scan beyond each original XCMS edge.

An explicit `refine_peak_boundaries_guarded(...)` helper now makes that policy
reusable. It returns the untouched core diagnostics plus guarded final bounds.
`guard_apex_outside` / `guard_no_overlap` require reference QC rejection. It does
not change the existing core default, model architecture, 16 attributes, concat
default or trained-model preprocessing.

The exploratory four-study audit favors the guard over the unrestricted core
for **reference construction when reliable upstream bounds exist**. The core's
median side error was 0.470 min, guard 0.057 min, native XCMS 0.066 min; guard
median IoU was 0.620. Seventeen of 473 guarded intervals exclude the core apex
and must be reviewed/rejected. This policy is not independent human truth and
may preserve an upstream boundary error. The actual-RT benchmark continues
using native XCMS `into`; none of its old results were rerun with the guard.

Replay this CPU-only policy audit from the cached raw EICs:

```powershell
python PeakTruthLab/scripts/rt_validation/compare_boundary_policies.py --audit-dir D:\ChromaPeakData\results\boundary_policy_audit --output-dir D:\ChromaPeakData\boundary_recheck
```

This uses upstream XCMS RT/bounds to predict and loads manual boundaries for
evaluation only after all new predictions finish. The model column is its
previously locked output, not a newly fitted comparison. This is an exploratory
audit of 473 jointly matched manual-positive peaks, not an independent retest or
an estimate of recall across all original features.
