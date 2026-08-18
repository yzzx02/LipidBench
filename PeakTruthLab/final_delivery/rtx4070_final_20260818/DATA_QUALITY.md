# Final dataset quality report

## Counts

| Item | Count |
|---|---:|
| Images / Seed candidates | 19,817 |
| PNG files | 19,817 |
| LabelMe JSON files | 19,817 |
| Seed true | 11,109 (56.06%) |
| Seed false | 8,708 (43.94%) |
| Images with at least one human box | 11,659 |
| Annotated peak instances | 18,915 |
| `True_Peak` shapes | 18,815 |
| `OUT_FIG` shapes | 100 |
| Mean annotated peaks per boxed image | 1.622 |
| Single-peak boxed images | 6,980 (59.87%) |
| Multi-peak boxed images | 4,679 (40.13%) |

The split contains 15,853 Train, 1,982 Val, and 1,982 Test images. The final merged data combine the protected, manually corrected old batch (15,317 images) with 4,500 newly reviewed samples. The old source is retained as provenance but the earlier experiment results are retired.

## Attributes

All master-table attributes are raw values: `SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`. Missing values remain explicit and are handled only by Train-fitted preprocessing inside each experiment.

## Verification

- portable-path issues: 0
- missing image/JSON files: 0
- content-hash mismatches: 0
- image ID, content hash, and duplicate-group overlap across Train/Val/Test: 0
- test CSV SHA-256: `b92aefb41be6dfdbbb025059873e7b5cbd3ce381c534d153b109fa7bd04de9c8`
- test JSONL SHA-256: `618abf3661ffb5a2bfe5a58b64ba444604353c0e27f678e52dcbd3a7f49b4461`

Detailed attribute distributions, duplicate groups, conflicts, exclusions, source/domain distributions, and the complete per-file SHA-256 manifest are in the dataset `audits` directory.
