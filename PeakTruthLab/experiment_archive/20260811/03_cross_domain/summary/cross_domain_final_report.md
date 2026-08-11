# Cross-domain Generalization Experiment: final 3-seed report

## Protocol

Naive concat only; seeds 20260725, 20260726, and 20260727; 15 epochs per seed. Attribute preprocessing was fitted from Train only. Val selected both checkpoints and thresholds. Seen-domain Control and External A/B/C were read only after selection was locked.

## Three-seed performance (mean ± sample SD)

| Condition | Detection F1 | Mean IoU | Seed BA | Seed AUROC | Seed AUPRC |
|---|---:|---:|---:|---:|---:|
| Seen-domain Control | 0.8460 ± 0.0317 | 0.8637 ± 0.0138 | 0.9261 ± 0.0096 | 0.9796 ± 0.0057 | 0.9974 ± 0.0008 |
| External A | 0.8081 ± 0.0192 | 0.8760 ± 0.0095 | 0.9085 ± 0.0235 | 0.9903 ± 0.0026 | 0.9787 ± 0.0049 |
| External B | 0.7900 ± 0.0073 | 0.8331 ± 0.0068 | 0.8663 ± 0.0033 | 0.9431 ± 0.0090 | 0.9498 ± 0.0142 |
| External C | 0.8787 ± 0.0053 | 0.8719 ± 0.0068 | 0.9541 ± 0.0086 | 0.9889 ± 0.0007 | 0.9948 ± 0.0007 |
| External Macro | 0.8256 ± 0.0065 | 0.8603 ± 0.0062 | 0.9096 ± 0.0115 | 0.9741 ± 0.0038 | 0.9744 ± 0.0062 |

External Macro is the unweighted mean of External A/B/C within each seed, followed by the three-seed mean and sample SD.

## Seen versus unseen domain gap

External Macro minus Seen is reported through Seen - Macro: Detection F1 0.0204 ± 0.0378, mean IoU 0.0034 ± 0.0081, Seed BA 0.0164 ± 0.0083, Seed AUROC 0.0055 ± 0.0024, Seed AUPRC 0.0230 ± 0.0055.

## Domain-specific interpretation

Seen-domain Control achieved Detection F1 0.8460 ± 0.0317, mean IoU 0.8637 ± 0.0138, Seed BA 0.9261 ± 0.0096, AUROC 0.9796 ± 0.0057, and AUPRC 0.9974 ± 0.0008.
External A (ST003127; seen instrument but unseen Study/chromatographic condition) showed a Seen-minus-A Detection F1 gap of 0.0379 ± 0.0508; localization remained strong because its mean-IoU gap was -0.0124 ± 0.0043.
External B (ST003941; unseen Orbitrap ID-X) had the largest adverse gap: Detection F1 0.0560 ± 0.0258, mean IoU 0.0306 ± 0.0139, Seed BA 0.0598 ± 0.0064, AUROC 0.0365 ± 0.0037, and AUPRC 0.0476 ± 0.0134.
External C (ST003514; seen Agilent 6545 with an unseen Poroshell EC-C18 combination) retained strong performance. Negative Seen-minus-C gaps for Detection F1 (-0.0327 ± 0.0368) and Seed BA (-0.0280 ± 0.0077) mean C exceeded Seen on average for those metrics.
External B had the lowest three-seed mean Detection F1 and was the lowest-F1 domain in two of three seeds; in seed 20260726, External A was slightly lower (0.7885 versus 0.7933). B had the lowest mean IoU, Seed BA, AUROC, and AUPRC in all three seeds. Thus its localization and Seed degradation was highly consistent, although the Detection F1 ranking and Precision-versus-Recall trade-off varied slightly by seed.

## External B failure analysis

External B remained the weakest domain on average: Detection F1 0.7900 ± 0.0073, mean IoU 0.8331 ± 0.0068, Seed BA 0.8663 ± 0.0033, AUROC 0.9431 ± 0.0090, and AUPRC 0.9498 ± 0.0142.
Detection Precision was 0.8102 ± 0.0412 and Recall was 0.7741 ± 0.0486; Recall was lower by 0.0361 on average, but the sign differed by seed. Seed sensitivity was 0.8889 and specificity was 0.8437; mean confusion counts were FP=100.7 and FN=107.3 per seed.
External B contained 38.9% empty windows versus 21.5% in Train and 7.0% in Seen. Its median box width was 21.5 px versus 51.1 px in Train and 64.9 px in Seen, indicating substantially narrower peaks and a plausible localization challenge.
The largest attribute median shifts relative to Train were: DZZ (+0.82 SD), ENT (-0.76 SD), SKEW (-0.50 SD), SNR (-0.49 SD), ZZ (+0.49 SD). These are descriptive associations, not causal or statistically significant claims.
External B uses the unseen Thermo Orbitrap ID-X Tribrid platform. Its lower detection localization and Seed ranking/classification performance are consistent with a combined instrument and feature-distribution shift. No B-specific threshold or fine-tuning was used.

## Integrity audit

Overall audit: PASS. Train∩Val source_file=0; Train∩Seen source_file=0; Val∩Seen source_file=0; Development∩External source_file=0; External Study∩Development Study=0. All three scaler files declare Train-only fitting, all training summaries declare test_manifest_used=false, and all selection locks prove Seen/External exclusion.

The three-seed trends, raw values, domain gaps, descriptive statistics, failure tables, and figures are stored beside this report.
