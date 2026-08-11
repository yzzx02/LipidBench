# Cross-domain generalization: Stage 1 report

> Status: metadata/split proposal complete; training not started.

## A. Formal dataset
- Windows: 15,317
- Source files: 29
- Studies: 11
- File-level condition domains: 17
- Instruments: 7
- Columns: 10

## B. Excel mapping
- Matched formal sources: 29/29
- Unmatched: 0
- Conflicts: 0
- Duplicate non-conflicting source keys: 2
- Excel-only sources without formal EIC windows: 4

## C. Recommended External domains
### External A = ST003127
- Instrument: Thermo Q Exactive Orbitrap
- Column: Imtakt Cadenza CD-C18 (150 × 2.1 mm, 3 µm)
- Source files / windows / true peaks: 2 / 994 / 469
- Seed positive / negative: 312 / 682
- Reason: unseen Study and chromatographic condition with the exact instrument retained in Development; unseen instrument-column combination; 994 windows, 469 boxes, Seed +/−=312/682; instrument_seen_elsewhere=True, column_seen_elsewhere=False, combo_seen_elsewhere=False

### External B = ST003941
- Instrument: Thermo Orbitrap ID-X Tribrid
- Column: Waters ACQUITY UPLC BEH C8 (100 × 2.1 mm, 1.7 µm)
- Source files / windows / true peaks: 3 / 1,610 / 1,241
- Seed positive / negative: 966 / 644
- Reason: unseen instrument with the column retained in Development; unseen instrument; 1610 windows, 1241 boxes, Seed +/−=966/644; instrument_seen_elsewhere=False, column_seen_elsewhere=True, combo_seen_elsewhere=False

### External C = ST003514
- Instrument: Agilent 6545 QTOF
- Column: Agilent InfinityLab Poroshell 120 EC-C18 (100 × 3 mm, 2.7 µm)
- Source files / windows / true peaks: 6 / 2,630 / 3,746
- Seed positive / negative: 1,989 / 641
- Reason: unseen instrument-column combination with the exact instrument retained in Development; unseen instrument-column combination; 2630 windows, 3746 boxes, Seed +/−=1989/641; instrument_seen_elsewhere=True, column_seen_elsewhere=False, combo_seen_elsewhere=False

## D. Proposed split
- Train windows: 7,274
- Val windows: 1,540
- External A windows: 994
- External B windows: 1,610
- External C windows: 2,630
- Locked original mixed-domain Test windows (unused): 1,269

## E. Leakage check
- Result: PASS
- Train ∩ Val source_file: 0
- Train ∩ External source_file: 0
- Val ∩ External source_file: 0
- External Study present in Train/Val: 0
- External B instrument present in Train/Val: 0
- External C instrument-column combination present in Train/Val: 0

No training was started. External A/B/C remain a proposal pending user confirmation.
