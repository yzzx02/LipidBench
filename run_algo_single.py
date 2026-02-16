"""Run one peak-picking algorithm on a single mzML.

This is a headless counterpart of the GUI "运行并加载" flow.
It runs the selected algorithm, writes outputs under results-dir/<algo>/,
then prints the produced feature table path.

Examples:
  python run_algo_single.py --algo pyopenms --mzml data/example.mzML --results-dir Results
  python run_algo_single.py --algo xcms --mzml data/example.mzML --results-dir Results
  python run_algo_single.py --algo asari --mzml data/example.mzML --results-dir Results
  python run_algo_single.py --algo msdial --msdial-xlsx Results/msdial_export.xlsx --results-dir Results
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path


def _resolve_input_mzml_files(mzml_path: Path | None, mzml_dir: Path | None) -> list[Path]:
    if mzml_dir is not None:
        if not mzml_dir.exists() or not mzml_dir.is_dir():
            raise FileNotFoundError("--mzml-dir must exist and be a directory")
        files = sorted(mzml_dir.glob("*.mzML"))
        if not files:
            raise FileNotFoundError("--mzml-dir has no .mzML files")
        return files
    if mzml_path is None or not mzml_path.exists():
        raise FileNotFoundError("--mzml is required and must exist")
    return [mzml_path]


def _run_one(*, algo: str, mzml_path: Path | None, mzml_dir: Path | None, results_dir: Path, msdial_xlsx: Path | None) -> Path:
    from lipidbench.utils.config_io import load_config

    config = load_config()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    algo = algo.strip().lower()

    if algo == "msdial":
        if msdial_xlsx is None or not msdial_xlsx.exists():
            raise FileNotFoundError("--msdial-xlsx is required for msdial")
        out_dir = results_dir / "msdial"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{msdial_xlsx.stem}_processed.csv"
        from lipidbench.utils.data_io import load_msdial_results

        load_msdial_results(msdial_xlsx.resolve(), out_file)
        return out_file

    input_files = _resolve_input_mzml_files(mzml_path, mzml_dir)

    with tempfile.TemporaryDirectory(prefix="tmp_mzml_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for p in input_files:
            shutil.copy2(p.resolve(), tmp_dir / p.name)

        if algo == "xcms":
            out_dir = results_dir / "xcms"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "xcms_features.csv"
            from lipidbench.runners.run_xcms import extract_xcms_params, run_xcms
            from lipidbench.utils.data_io import load_xcms_results

            params = extract_xcms_params(config)
            run_xcms(input_dir=tmp_dir, output_file=out_file, **params)
            load_xcms_results(out_file)
            return out_file

        if algo == "pyopenms":
            out_dir = results_dir / "pyopenms"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "pyopenms_features.csv"
            from lipidbench.runners.run_pyopenms import extract_pyopenms_params, run_pyopenms
            from lipidbench.utils.data_io import load_pyopenms_results

            params = extract_pyopenms_params(config)
            run_pyopenms(input_dir=tmp_dir, output_file=out_file, **params)
            load_pyopenms_results(out_file, input_dir=tmp_dir, **params)
            return out_file

        if algo == "asari":
            cfg = copy.deepcopy(config)
            cfg.setdefault("paths", {})
            cfg["paths"]["input_dir"] = str(tmp_dir)
            cfg["paths"]["asari_output"] = str((results_dir / "asari").resolve())
            from lipidbench.runners.run_asari import run_asari_pipeline

            run_asari_pipeline(cfg)

            asari_dir = results_dir / "asari"
            preferred = asari_dir / "preferred_Feature_table.csv"
            full = asari_dir / "full_Feature_table.csv"
            if preferred.exists():
                return preferred
            if full.exists():
                return full
            raise FileNotFoundError(f"Asari feature table not found: {preferred} or {full}")

    raise ValueError(f"Unknown --algo: {algo}")


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(config)

    cfg.setdefault("parameters", {})

    # xcms
    cfg["parameters"].setdefault("xcms", {})
    cfg["parameters"]["xcms"].setdefault("peak_picking", {})
    xcms_peak = cfg["parameters"]["xcms"]["peak_picking"]
    if args.xcms_ppm is not None:
        xcms_peak["ppm"] = float(args.xcms_ppm)
    if args.xcms_noise is not None:
        xcms_peak["noise"] = float(args.xcms_noise)
    if args.xcms_sn is not None:
        xcms_peak["snthresh"] = float(args.xcms_sn)
    if args.xcms_minwidth is not None or args.xcms_maxwidth is not None:
        minw = float(args.xcms_minwidth) if args.xcms_minwidth is not None else float(xcms_peak.get("peakwidth", [5, 50])[0])
        maxw = float(args.xcms_maxwidth) if args.xcms_maxwidth is not None else float(xcms_peak.get("peakwidth", [5, 50])[1])
        xcms_peak["peakwidth"] = [minw, maxw]

    # pyopenms
    cfg["parameters"].setdefault("pyopenms", {})
    pyopenms_cfg = cfg["parameters"]["pyopenms"]
    pyopenms_cfg.setdefault("peak_picking", {})
    pyopenms_peak = pyopenms_cfg["peak_picking"]
    if args.pyopenms_ppm is not None:
        pyopenms_peak["mz_tol"] = float(args.pyopenms_ppm)
    if args.pyopenms_noise is not None:
        pyopenms_peak["noise"] = float(args.pyopenms_noise)
    if args.pyopenms_sn is not None:
        pyopenms_peak["sn"] = float(args.pyopenms_sn)
    if args.pyopenms_min_fwhm is not None:
        pyopenms_peak["min_fwhm"] = float(args.pyopenms_min_fwhm)
    if args.pyopenms_max_fwhm is not None:
        pyopenms_peak["max_fwhm"] = float(args.pyopenms_max_fwhm)

    # asari
    cfg["parameters"].setdefault("asari", {})
    asari_cfg = cfg["parameters"]["asari"]
    if args.asari_ppm is not None:
        asari_cfg["ppm"] = float(args.asari_ppm)
    if args.asari_min_intensity_threshold is not None:
        asari_cfg["min_intensity_threshold"] = float(args.asari_min_intensity_threshold)
    if args.asari_mode is not None:
        asari_cfg["mode"] = str(args.asari_mode)
    if args.asari_autoheight is not None:
        asari_cfg["autoheight"] = bool(args.asari_autoheight)

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["asari", "pyopenms", "xcms", "msdial"], help="Algorithm to run")
    parser.add_argument("--mzml", type=Path, help="Input mzML (required for asari/pyopenms/xcms)")
    parser.add_argument("--mzml-dir", type=Path, help="Input mzML directory for multi-file run")
    parser.add_argument("--results-dir", type=Path, required=True, help="Output base directory")
    parser.add_argument("--msdial-xlsx", type=Path, help="MS-DIAL exported xlsx (required for msdial)")
    parser.add_argument("--xcms-ppm", type=float)
    parser.add_argument("--xcms-noise", type=float)
    parser.add_argument("--xcms-sn", type=float)
    parser.add_argument("--xcms-minwidth", type=float)
    parser.add_argument("--xcms-maxwidth", type=float)

    parser.add_argument("--pyopenms-ppm", type=float)
    parser.add_argument("--pyopenms-noise", type=float)
    parser.add_argument("--pyopenms-sn", type=float)
    parser.add_argument("--pyopenms-min-fwhm", type=float)
    parser.add_argument("--pyopenms-max-fwhm", type=float)

    parser.add_argument("--asari-ppm", type=float)
    parser.add_argument("--asari-min-intensity-threshold", type=float)
    parser.add_argument("--asari-mode", choices=["pos", "neg"])
    parser.add_argument("--asari-autoheight", choices=["true", "false"])
    args = parser.parse_args()

    # normalize bool-like CLI argument
    if isinstance(args.asari_autoheight, str):
        args.asari_autoheight = args.asari_autoheight.lower() == "true"

    from lipidbench.utils.config_io import load_config

    _ = _apply_overrides(load_config(), args)

    # Reuse _run_one behavior; _run_one will read config again via downstream runners,
    # so apply overrides by temporarily patching config file-independent runner parameters
    # through environment-backed argparse-style passthrough when needed.
    # Here we pass overrides by monkeypatching runner config inside this process.
    config = load_config()
    config = _apply_overrides(config, args)

    # Inline _run_one logic with overridden config
    def _run_one_with_config(*, algo: str, mzml_path: Path | None, mzml_dir: Path | None, results_dir: Path, msdial_xlsx: Path | None, config: dict) -> Path:
        results_dir = results_dir.resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        algo = algo.strip().lower()

        if algo == "msdial":
            if msdial_xlsx is None or not msdial_xlsx.exists():
                raise FileNotFoundError("--msdial-xlsx is required for msdial")
            out_dir = results_dir / "msdial"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{msdial_xlsx.stem}_processed.csv"
            from lipidbench.utils.data_io import load_msdial_results

            load_msdial_results(msdial_xlsx.resolve(), out_file)
            return out_file

        input_files = _resolve_input_mzml_files(mzml_path, mzml_dir)

        with tempfile.TemporaryDirectory(prefix="tmp_mzml_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for p in input_files:
                shutil.copy2(p.resolve(), tmp_dir / p.name)

            if algo == "xcms":
                out_dir = results_dir / "xcms"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "xcms_features.csv"
                from lipidbench.runners.run_xcms import extract_xcms_params, run_xcms
                from lipidbench.utils.data_io import load_xcms_results

                params = extract_xcms_params(config)
                run_xcms(input_dir=tmp_dir, output_file=out_file, **params)
                load_xcms_results(out_file)
                return out_file

            if algo == "pyopenms":
                out_dir = results_dir / "pyopenms"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / "pyopenms_features.csv"
                from lipidbench.runners.run_pyopenms import extract_pyopenms_params, run_pyopenms
                from lipidbench.utils.data_io import load_pyopenms_results

                params = extract_pyopenms_params(config)
                run_pyopenms(input_dir=tmp_dir, output_file=out_file, **params)
                load_pyopenms_results(out_file, input_dir=tmp_dir, **params)
                return out_file

            if algo == "asari":
                cfg = copy.deepcopy(config)
                cfg.setdefault("paths", {})
                cfg["paths"]["input_dir"] = str(tmp_dir)
                cfg["paths"]["asari_output"] = str((results_dir / "asari").resolve())
                from lipidbench.runners.run_asari import run_asari_pipeline

                run_asari_pipeline(cfg)

                asari_dir = results_dir / "asari"
                preferred = asari_dir / "preferred_Feature_table.csv"
                full = asari_dir / "full_Feature_table.csv"
                if preferred.exists():
                    return preferred
                if full.exists():
                    return full
                raise FileNotFoundError(f"Asari feature table not found: {preferred} or {full}")

        raise ValueError(f"Unknown --algo: {algo}")

    feature_path = _run_one_with_config(
        algo=args.algo,
        mzml_path=args.mzml,
        mzml_dir=args.mzml_dir,
        results_dir=args.results_dir,
        msdial_xlsx=args.msdial_xlsx,
        config=config,
    )
    print(str(feature_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
