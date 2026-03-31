# RT Boundary Refinement Module

## Goal
- Recompute `RTmin/RTmax` from `mz + RT` directly on the extracted EIC trace
- Use one consistent rule set before training and before inference
- Make boundary behavior closer to mainstream LC-MS software logic instead of trusting raw exported bounds verbatim

## Main Idea
The module in `lipidbench/utils/rt_boundary_refiner.py` is not a byte-for-byte reimplementation of any single vendor or open-source package. Instead, it follows the common peak-boundary logic shared by MS-DIAL / XCMS / OpenMS:

1. Smooth the chromatographic trace first
2. Estimate local baseline and local noise
3. Use the feature `RT` as the anchor and search a local apex nearby
4. Walk left/right from the apex until the trace becomes baseline-like
5. Expand a little further to the nearest local minimum when appropriate
6. If the nearest local minimum is followed by a clear rebound farther away, stop at that minimum
7. If the resulting boundary is obviously too wide and contains too much baseline, shrink back to the apex-centered core boundary

## Why This Matches Mainstream Software Better
- **MS-DIAL-like part**
  - linear weighted moving average style smoothing
  - use of local amplitude / derivative thresholds to separate signal from baseline
  - stop near local minima rather than blindly keeping a wide tail
- **XCMS / centWave-like part**
  - local noise-aware, apex-centered boundary determination
  - boundary is tied to the candidate peak around the provided `RT`, not to a manually dragged box
- **OpenMS-like part**
  - robust against noisy chromatograms
  - avoids treating a very long flat baseline as valid peak width

## Special Cases Covered
### 1. Noisy peaks
- Smooth with repeated LWMA
- Estimate baseline/noise from a trimmed local region around the apex
- Require multiple consecutive baseline-like scans before confirming the boundary

### 2. Wrong box / shifted manual boundary
- If old `RTmin/RTmax` are wrong or `RT` is not inside them, the module does not trust the old boundary
- It recenters from the feature `RT` and finds the apex again from the trace

### 3. Boundary too wide and includes long baseline
- First compute an apex-centered core boundary
- Then optionally extend to a nearby valley
- If the extended boundary becomes much wider than the core and both edges are already near baseline, shrink back to the core

## Files
- Core module: `lipidbench/utils/rt_boundary_refiner.py`
- Dataset/prediction preprocessing entrypoint: `PeakTruthLab/scripts/data_prep/refine_feature_rt_bounds.py`
- Shared batch worker: `PeakTruthLab/scripts/annotation/recompute_rt_bounds_and_attrs.py`
- LabelMe-assisted gold-boundary correction only: `PeakTruthLab/scripts/annotation/recompute_rt_bounds_from_labelme.py`

## Recommended Usage
### Training-time preprocessing
- Run the refinement module on all candidate features before final attribute calculation
- Recompute the 13 peak attributes only after refined `RTmin/RTmax` are available
- Default extraction tolerance is `15 ppm`, matching the current high-resolution MS setup

Example:

```bash
python PeakTruthLab/scripts/annotation/recompute_rt_bounds_and_attrs.py \
  --input-csv PeakTruthLab/datasets/feature_table_final_10000.csv \
  --output-csv PeakTruthLab/datasets/feature_table_final_10000.csv \
  --report-csv PeakTruthLab/results/rt_bounds_recompute_report.csv \
  --mz-tolerance 10 \
  --tolerance-unit ppm \
  --search-half-window-min 0.35 \
  --local-half-window-min 1.0 \
  --smooth-window-scans 5 \
  --smooth-passes 2 \
  --max-expand-scans 18 \
  --max-expand-min 0.35 \
  --boundary-confirm 2 \
  --oversize-factor 1.8 \
  --update-rt-mode when_outside_bounds
```

### Inference-time preprocessing
- Given a new candidate feature from XCMS / pyOpenMS:
  - extract EIC
  - refine boundary using the same no-JSON module
  - recompute attributes
  - render image
  - send refined sample into the classifier
- In inference, do **not** depend on LabelMe JSON.
- Use the candidate `RT + RTmin + RTmax` from upstream software as the initial hint only, then let the refinement module decide whether to extend, shrink, or recenter.

## Dataset Workflow vs Prediction Workflow
### Current dataset preprocessing
- If a sample has manual LabelMe correction and you want a gold-standard boundary, use `recompute_rt_bounds_from_labelme.py`
- After that, the no-JSON RT refinement logic is still the deployable boundary rule

### Prediction preprocessing
- Prediction has no JSON
- Therefore prediction should use `refine_feature_rt_bounds.py` directly on candidate features exported by XCMS / pyOpenMS
- This keeps deployment logic consistent with the boundary rules used to compute final attributes

## Test Coverage
- Tests live in `tests/test_rt_boundary_refiner.py`
- Covered synthetic scenarios:
  - noisy peak
  - RT outside old bounds
  - remote stronger peak should not steal local apex
  - overwide boundary should shrink
  - no local signal should return `zero_apex`

## Key Rule
Training and inference must use the same boundary refinement policy. Otherwise the image branch and attribute branch will see different data distributions.
