# Cross-domain generalization: single-seed result

- Model: Naive concat
- Random seed: 20260725
- Best detection epoch: 9
- Best Seed epoch: 15
- Val-selected detection threshold: 0.5
- Val-selected Seed threshold: 0.513
- External data were evaluated only after both checkpoints and thresholds were locked.

| Condition | Detection F1 | Mean IoU | Seed BA | Seed AUROC | Seed AUPRC |
|---|---:|---:|---:|---:|---:|
| External A | 0.8090 | 0.8801 | 0.9283 | 0.9922 | 0.9799 |
| External B | 0.7951 | 0.8265 | 0.8701 | 0.9425 | 0.9445 |
| External C | 0.8802 | 0.8759 | 0.9611 | 0.9881 | 0.9940 |
| External macro average | 0.8281 | 0.8608 | 0.9198 | 0.9743 | 0.9728 |

Training time: 48.6 min
Peak CUDA allocated memory: 3.33 GiB

External macro average is the unweighted arithmetic mean of the three separately evaluated domains. No pooled-only External metric is used.
