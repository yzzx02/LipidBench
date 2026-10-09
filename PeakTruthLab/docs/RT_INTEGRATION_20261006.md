# RT result integration review — 2026-10-06

## GitHub basis

- PR #9 was already merged into `main` as
  `660ddbea5192f795f7236aedf0c2ffe1421eb572`. This integration starts from that
  exact commit, retaining ChromaPeak naming, maintained entry points, 16
  attributes and the current naive-concat defaults.
- PR #5 is still open and conflicts with current main. Its older 4500-transfer
  and attribute-migration work is not needed for this RT replay; the later final
  dataset/current attribute implementation are already maintained on main.
  Retired training entry points are not restored by this integration.
- The original local checkout has an unrelated uncommitted locked-target
  evaluator change. Work was performed in a separate Git worktree and that
  local change was preserved.

## Algorithm decision

PR #9's core boundary changes add finite/increasing-axis validation, a finite
hint check and length-preserving smoothing on short traces. They retain the
normal-trace algorithm. An exploratory rerun on the earlier 473 paired raw EICs
found exactly identical apex/left/right outputs before and after PR #9.

The recovery experiment's additional one-MS1-scan outward-extension cap was
previously implemented in experiment scripts. It is now available through the
explicit `refine_peak_boundaries_guarded` API; the original core signature and
defaults remain intact. Measured median side errors: unrestricted core 0.4702
min, guarded core 0.0574 min, native XCMS 0.0657 min. The guarded policy is the
preferred available traditional reference-construction option on this selected
cohort, assuming checked upstream bounds. Seventeen clipped intervals exclude
the core apex and are reported as QC failures. This does not establish an
independent universal optimum, and it does not change trained-model inputs.

The actual-RT benchmark uses full-spectrum XCMS centWave/native `into` plus
fixed-prior matching, without boundary refinement or alignment. Its historical
224,000 results are preserved. The model still uses its historical image-only
checkpoint; substituting the default concat checkpoint is rejected by SHA-256.

## Transfer and checks

Summaries, parameters, manifests and paper deliverables are now machine-local
under `<CHROMAPEAK_LOCAL_ROOT>/artifacts/reproduction/PeakTruthLab/final_delivery/`.
Full per-peak tables, original 16 mzML files, raw predictions/logs, frozen
references/history, historical code, cached boundary-audit EICs, exact checkpoint
and matching R runtime are Release assets. Figures are excluded. Archive CRCs
and SHA-256 are verified before upload; Release digests are checked after upload.

The portable runner relocates data paths while verifying original bytes;
reference values and their hashes remain fixed. It regenerates actual RT shifts,
checks spectral binary invariance and gates the full experiment on a successful
baseline. NMS, RoIAlign, checkpoint loading, EIC batch extraction and software/
source/device fingerprints are checked. Replay results are compared against the
original RTX4070 summaries rather than overwriting them.

A transfer smoke test exposed an implicit Matplotlib font dependency: the old
annotation-helper import changed worker rcParams. The replay renderer now sets
the exact historical Arial/sans-serif policy explicitly in each worker. This
preserves EIC pixels and prevents different multiprocessing entry points from
changing model inputs. No detector threshold, matching rule or weight was tuned.

Verification scope: all 69 repository tests; full 473-EIC CPU boundary-policy
replay; actual model inference on two baseline EICs and two actual −0.5 min EICs;
one-file native XCMS replay; strict checkpoint/ops preflight on RTX4070 Super.
The full 224,000 inference records were not recomputed for this transfer, and the
RX 9070 XT itself has not yet been tested. That run is the purpose of the supplied
replay entry point and per-hardware comparison outputs.
