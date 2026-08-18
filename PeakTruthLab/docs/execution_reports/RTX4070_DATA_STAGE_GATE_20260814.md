# RTX 4070 data-stage gate report (2026-08-14)

## Outcome

Execution followed `PeakTruthLab/docs/RTX4070_MERGE_AND_SPLIT_PROMPT.md` and stopped at the 16-attribute merge gate. The protected local PeakTruthLab v2 dataset and the newly released 4,500-sample package both passed their applicable integrity checks. The merge, split, smoke test, ablation training, main-test evaluation, and LODO experiments were **not started**, because the local old-final dataset has only 13 attributes and the original mzML files needed to compute `SYM`, `MOD`, and `EDGE` are absent from this workstation and the repository releases.

No file in the protected old-final dataset was modified, moved, or overwritten.

## 1. Protected local old-final dataset

Authoritative path:

`D:\CODE\LipidBench\PeakTruthLab\results\paper_final_reviewed_20260725\dataset_release\PeakTruthLab-dataset-v2`

Why this is the old-final dataset rather than the earlier GitHub copy:

- Main image count: 15,317.
- Manifest rows: Train 10,525; Val 2,507; Test 2,285.
- The 150 LabelMe review JSON files are byte-identical to the workstation review directory (`REVIEW_JSON_DIFF_COUNT=0`).
- The archived correction report records 150 reviewed samples, 66 manually changed samples, 52 boxes added, 23 boxes removed, and 29 Seed-label changes (21 negative-to-positive; 8 positive-to-negative).
- Train and Test manifests are byte-identical to the parent v1 package, while Val has the corrected v2 hash. This matches the documented Val-only review workflow.
- The v2 archive passed its recorded SHA256 check.

Core hashes:

| Item | SHA256 |
|---|---|
| `PeakTruthLab-dataset-v2.zip` | `070A1DA29179F9B57B9A155D6EF88F29AC21C2287C3B72286C5A0EB12FDB2DE2` |
| `manifests/train.jsonl` | `8163D3D4F57C322AAA6CF2DC7F45C495A52B84D654BF4354D6ADBE29309036A7` |
| `manifests/val.jsonl` | `E3B461CB74D4A2743D74C94771133F3D7B29E8B50D54E016BE18C02BC933C76C` |
| `manifests/test.jsonl` | `16C2C80106C688EF1E35E493FEA06129EBD168CBFD28CA6EA293EF577D4B5A75` |

Representative protected-image hashes:

| Image | SHA256 |
|---|---|
| `0001_MAY_RoCI-StEM_CP-287__F10001.png` | `AD33A3B978D8CCE35BB762FBB4AD4B0C1082360F05C82B0E7A7FC447E36EE56B` |
| `D03P_POS__F1233.png` | `8927B86B2D334AA4F3DF2FF07C0CB6D097C3DFC85329DA08F8593FB40CA08A7A` |
| `WTHFD_mixpos__F9990.png` | `D3CFC07ACF6132DE4D84F24CC3C591D9A95FA9E7385EE49FF9B6A33C2C1C1743` |

## 2. New 4,500-sample release validation

Downloaded asset:

`D:\CODE\downloads\PeakTruthLab_RTX4070_20260814\PeakTruthLab_manual_negative_4500_v2_20260814.zip`

Isolated extraction root:

`D:\CODE\downloads\PeakTruthLab_RTX4070_20260814\extracted\manual_negative_4500_v2_staging`

Release SHA256:

`F89632310064E5F92A484AB7FD75A05ABF3DC55631F7077B2C9A328D4F0CB409` (exact match)

Independent checks:

| Check | Result |
|---|---:|
| Corrected PNG files | 4,500 |
| Corrected LabelMe JSON files | 4,500 |
| PNG dimensions | all 480 x 480 |
| Invalid JSON | 0 |
| Seed rows / unique image IDs / unique feature IDs | 4,500 / 4,500 / 4,500 |
| Peak rows / unique peak IDs | 335 / 335 |
| `True_Peak` shapes | 328 |
| `OUT_FIG` shapes | 7 |
| Non-finite Seed attribute values (16 columns) | 0 |
| Non-finite peak-instance attribute values (16 columns) | 0 |

For `HARDNEG3000_20260813READY__PLAT__D03P_POS__F0108`, the restored Seed rectangle in the LabelMe JSON exactly equals the Seed-table bounds `[221.1275313678, 84.35990338164245, 269.2010647012018, 403.75925925925924]`.

## 3. Blocking condition

The protected old-final Seed and peak tables expose the original 13 attributes only:

`SNR,CV,GS,TPAS,H2B,ZZ,DZZ,PCC,SKEW,DENT,DM,ENT,JAG`

The execution plan requires adding:

`SYM,MOD,EDGE`

The checked-in implementation computes those three values from the original EIC window and original apex index. It therefore requires the source mzML files; substituting PNG-derived approximations, zero filling, global imputation, or copied values from another sample would violate the frozen data protocol.

The old-final table contains 15,317 rows from 29 unique mzML files. Read-only searches found zero mzML files under the relevant locations on C:, D:, E:, and F:. The two GitHub releases contain the portable v1 image package and the new 4,500-sample package, but no source mzML. At least one required source, `frag1_pos20_1.mzML` (1,000 old-final samples), is self-collected and cannot be recovered from a public study.

Required source inventory:

| mzML file | Old-final rows |
|---|---:|
| `0001_MAY_RoCI-StEM_CP-287.mzML` | 465 |
| `0021_MAY_ROCI-StEM_HN_575.mzML` | 502 |
| `060-0145-005_017.mzML` | 504 |
| `060-0145-006_018.mzML` | 515 |
| `20180321_S00033936_P.mzML` | 749 |
| `20180323_S00033882_N.mzML` | 683 |
| `6545A_20230727_-_57_18.mzML` | 521 |
| `6545A_20230729_+_07_9.mzML` | 519 |
| `6545_20201012_14_+_QC 01.mzML` | 512 |
| `6545_20201012_22_+_QC 03.mzML` | 515 |
| `AG-88-11_r12-.mzML` | 520 |
| `AG-88-11_r2+.mzML` | 516 |
| `Blood-15V.mzML` | 523 |
| `Blood-30V.mzML` | 520 |
| `D03P_POS.mzML` | 514 |
| `D04P_NEG.mzML` | 302 |
| `D06P_POS.mzML` | 517 |
| `D07P_NEG.mzML` | 284 |
| `HepG2-30V.mzML` | 524 |
| `HetCD_mixneg.mzML` | 506 |
| `NIST_Full scan_1_POS.mzML` | 502 |
| `NIST_Full scan_2_NEG.mzML` | 511 |
| `Sphingolipid-BTSPT-Cell-pellet_1.mzML` | 533 |
| `Sphingolipid-BTSPT-OMV_1.mzML` | 535 |
| `Sphingolipid-BTWT-Cell-pellet_1.mzML` | 542 |
| `Urine-15V.mzML` | 486 |
| `Urine-30V.mzML` | 509 |
| `WTHFD_mixpos.mzML` | 488 |
| `frag1_pos20_1.mzML` | 1,000 |

## 4. Required recovery action

Restore the original 29 mzML files, preserving their exact file contents, under a stable local root (the historical layout was `data/ceshiji`, with some supplementary files under `data/raw_mzML`). At minimum, `frag1_pos20_1.mzML` must come from the workstation/user backup; the public-study files should preferably also be restored from the same converted mzML copies to avoid conversion-pipeline differences.

After restoration, resume from the current branch and rerun the merge gate. The next allowed action is to compute only `SYM`, `MOD`, and `EDGE` for the protected old-final rows while preserving the existing 13 values and all manually corrected boundaries. No split or training should begin before this gate passes.
