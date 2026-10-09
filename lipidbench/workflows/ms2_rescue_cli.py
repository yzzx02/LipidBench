from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .ms2_rescue import RescueOptions, read_identification_table, rescue_ms2_peaks


def main(argv=None):
    parser = argparse.ArgumentParser(description="MS2-guided missing-feature rescue (no differential analysis)")
    parser.add_argument("--table", type=Path, help="CSV/TSV/XLSX identification table")
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--write-template", type=Path, help="Write an example CSV and exit")
    parser.add_argument("--column-map", type=Path, help="JSON mapping canonical column names to table headers")
    parser.add_argument("--mzml-root", type=Path, help="Base for relative mzML paths (default: table directory)")
    parser.add_argument("--detection-checkpoint", type=Path)
    parser.add_argument("--classification-checkpoint", type=Path, help="Released best_seed.pt")
    parser.add_argument("--preprocessor", type=Path, help="Matching train-fitted attribute_preprocessing.json")
    parser.add_argument("--selection", type=Path, help="Val selection_before_test.json supplies both thresholds")
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--classification-threshold", type=float)
    parser.add_argument("--rt-unit", choices=["min", "sec"], default="min")
    parser.add_argument("--window-min", type=float, default=2.0)
    parser.add_argument("--mz-tolerance-ppm", type=float, default=10.0)
    parser.add_argument("--ms2-rt-tolerance-min", type=float, default=0.2)
    parser.add_argument("--method", choices=["nearest", "window_sum"], default="nearest")
    parser.add_argument("--min-scans", type=int, default=3)
    parser.add_argument("--max-extension-scans", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.write_template:
        if args.write_template.exists():
            parser.error("template file already exists")
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {"target_id": "annotation_1", "sample_id": "sample_1", "mzml_path": "sample_1.mzML",
             "precursor_mz": 760.5851, "ms2_rt": 5.20, "has_ms1": True, "has_ms2": True},
            {"target_id": "annotation_2", "sample_id": "sample_1", "mzml_path": "sample_1.mzML",
             "precursor_mz": 734.5694, "ms2_rt": 6.15, "has_ms1": False, "has_ms2": True},
        ]).to_csv(args.write_template, index=False, encoding="utf-8-sig")
        print(f"Template: {args.write_template.resolve()}")
        return
    for name in ("table", "output", "detection_checkpoint", "classification_checkpoint", "preprocessor"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else {}
    detection_threshold = args.detection_threshold
    classification_threshold = args.classification_threshold
    if detection_threshold is None:
        detection_threshold = selection.get("detection", {}).get("selected_score_threshold")
    if classification_threshold is None:
        classification_threshold = selection.get("seed", {}).get("selected_probability_threshold")
    if detection_threshold is None or classification_threshold is None:
        parser.error("provide --selection or both explicit thresholds; thresholds are not guessed")
    options = RescueOptions(detection_threshold=detection_threshold, classification_threshold=classification_threshold,
                            rt_unit=args.rt_unit, window_min=args.window_min, mz_tolerance_ppm=args.mz_tolerance_ppm,
                            method=args.method, ms2_rt_tolerance_min=args.ms2_rt_tolerance_min,
                            min_scans=args.min_scans, max_extension_scans=args.max_extension_scans)
    mapping = json.loads(args.column_map.read_text(encoding="utf-8")) if args.column_map else None
    from .ms2_rescue_inference import TorchRescuePredictor
    predictor = TorchRescuePredictor(args.detection_checkpoint, args.classification_checkpoint,
                                    args.preprocessor, device=args.device, batch_size=args.batch_size)
    result = rescue_ms2_peaks(read_identification_table(args.table), predictor, args.output, options,
                             base_dir=args.mzml_root or args.table.resolve().parent, column_map=mapping)
    result.summary["model"] = predictor.provenance
    result.summary["identification_table"] = str(args.table.resolve())
    result.write(args.output)
    print(json.dumps({key: result.summary[key] for key in ("input_target_rows", "unique_rescued_peaks", "status_counts")},
                     ensure_ascii=False, indent=2))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
