# Three-seed stability experiment

All values are mean ± sample standard deviation across random seeds 20260725, 20260726 and 20260727. Hyperparameters and data splits were locked. Checkpoints and thresholds were selected on Validation data; the three Gated fusion Test runs started only after all three Gated training runs completed and all Validation thresholds were locked. Test results were not used to adjust any parameter.

| Condition | Detection F1 | Mean IoU | Seed balanced accuracy | Seed AUROC |
|---|---:|---:|---:|---:|
| Image only | 0.8608 ± 0.0045 | 0.8696 ± 0.0060 | 0.9651 ± 0.0019 | 0.9938 ± 0.0009 |
| Attributes only | 0.8521 ± 0.0103 | 0.8777 ± 0.0016 | 0.9520 ± 0.0033 | 0.9901 ± 0.0006 |
| Naive concat | 0.8613 ± 0.0060 | 0.8765 ± 0.0029 | 0.9663 ± 0.0058 | 0.9954 ± 0.0014 |
| Gated fusion | 0.8502 ± 0.0083 | 0.8727 ± 0.0065 | 0.9652 ± 0.0048 | 0.9947 ± 0.0013 |

The standard deviations describe between-training-run variability for these three fixed seeds; they are distinct from the earlier image-level bootstrap confidence intervals.
