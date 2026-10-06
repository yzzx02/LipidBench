# ChromaPeak: RT validation delivery

Measured experiment: **2026-09-30**. Integration/transfer audit: **2026-10-06**.
Based on merged PR #9 (`660ddbea5192f795f7236aedf0c2ffe1421eb572`). The project
name, current 16 attributes and naive-concat defaults are retained.

## Actual mzML RT translation

16 original CommercialQstdPooled mzML, 1,000 features detected by both methods
in all 16 baseline files, seven signed shifts and two methods: **224,000 rows**.
Original fixed m/z/RT priors do not move. Only actual spectrum/TIC time
coordinates move; spectral binaries stay identical. No RT alignment/correction,
grouping, gap filling, retraining or shift-specific tuning is applied.

XCMS 4.10.1: full-spectrum centWave followed by fixed-prior matching within
10 ppm and ±1 min. Parameters: ppm 5, peakwidth 5–50 s, noise 3000, SN 5,
prefilter 3/3000, mzdiff 0.001, wMean and native `into`. The original
image-only Faster R-CNN checkpoint uses 2 min EICs and score ≥0.5. Both select
the valid candidate apex closest to the unchanged prior.

| Actual shift/min | XCMS Recall | Model Recall | XCMS accurate quantification | Model accurate quantification |
|---:|---:|---:|---:|---:|
| -0.5 | 96.24% | 96.08% | 96.24% | 95.99% |
| -0.2 | 98.58% | 97.49% | 98.58% | 97.36% |
| -0.1 | 99.94% | 98.44% | 99.94% | 98.29% |
| 0 | 100% | 100% | 100% | 100% |
| +0.1 | 99.91% | 98.68% | 99.90% | 98.54% |
| +0.2 | 98.83% | 97.84% | 98.82% | 97.71% |
| +0.5 | 97.26% | 96.37% | 97.25% | 96.24% |

Correct detection: selected apex within 10 s of the frozen per-file original
apex + shift and interval containing that expected apex. Accurate quantification
additionally requires absolute own-baseline area error <20%; all targets remain
in the denominator. Areas use intensity×seconds. These are method-specific
stability references, not independent absolute concentration truth.

XCMS native peak counts/masses/areas translate unchanged in all 96 nonzero
file×shift runs. Most failures are wrong nearest-prior selection, rather than
native peak disappearance. Model target-candidate recall is 99.46%–99.77% before
selection; XCMS candidate recall is 100%. Both median area errors are essentially
zero; mean error ranges are XCMS 0.05%–2.72%, model 1.11%–3.16%.
The results do **not** demonstrate an RT-robustness advantage for the model.

The baseline-positive cohort is constructed jointly by these algorithms, not
an independent manually annotated gold standard. Two baseline coeluting mass
confusions were replaced using baseline-only identity QC; 998 references remain
unchanged and the new two were frozen before their target predictions/matching.
XCMS full-spectrum detection was already available for the added targets.
The maximum resulting metric change is 0.04375 percentage points; initial freeze
and extension history are in the full result archive. No shifted outcomes were
used to choose replacements.

## Boundary policy comparison

On the earlier 473 jointly matched manual-positive peaks in four studies:

| Policy | Median side error/min | Median boundary IoU |
|---|---:|---:|
| Native XCMS | 0.0657 | 0.567 |
| Old core / PR #9 core | 0.4702 | 0.240 |
| Core + existing one-scan extension guard | 0.0574 | 0.620 |
| Previously locked model boundaries | 0.0076 | 0.937 |

The core's normal-trace outputs are identical across versions. PR #9's input
safety fixes remain. The explicit guard is preferable here for automatic
reference construction using checked upstream bounds, but 17 capped intervals
exclude the selected apex and must be reviewed/rejected. Existing core defaults
are preserved to avoid changing released model preprocessing. This exploratory
audit does not reclassify the original actual-RT results or establish independent
cross-study generalization. The same paired cohort was selected from earlier
locked test outputs; `locked_model` is not a fresh inference claim.

## Files and transfer

The small summaries, parameters, frozen-reference hashes and input/checkpoint
manifests are committed here. The full per-peak tables, raw candidates/logs,
freeze history and historical code for actual shifts, 490-target prior recovery
and the four-study boundary comparison are in
[the Release](https://github.com/yzzx02/LipidBench/releases/tag/chromapeak-rt-validation-20260930).
All figures are omitted from the transfer payload.

Use [the portable replay guide](../../scripts/rt_validation/README.md) on the new
workstation. It verifies original input hashes, the exact historical checkpoint,
NMS/RoIAlign, XCMS version and baseline compatibility before a full run, then
reports hardware differences against the saved RTX4070 summaries. GPU inference
has been smoke-checked on the current RTX4070; the RX 9070 XT run remains to be
performed on that workstation.
