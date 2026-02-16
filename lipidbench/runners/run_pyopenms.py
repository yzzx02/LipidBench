import os
import sys
from pathlib import Path

from typing import Any

from lipidbench.utils.config_io import get_base_dir, _resolve_path
from lipidbench.utils.data_io import load_pyopenms_results


def _get_oms():
    try:
        import pyopenms as oms  # type: ignore

        return oms
    except Exception as e:
        raise ImportError(
            "pyopenms 导入失败（常见原因：Python 版本不匹配 / 缺少 VC++ 运行库 / 环境损坏）。\n"
            "建议：切换到已安装 pyopenms 的解释器，并检查 VC++ 运行库。\n"
            f"当前 Python: {sys.version.split()[0]}\n"
            f"当前解释器: {sys.executable}\n"
            f"原始错误: {e}"
        )


def align_features(feature_maps):
    oms = _get_oms()
    ref_index = feature_maps.index(sorted(feature_maps, key=lambda x: x.size())[-1])
    aligner = oms.MapAlignmentAlgorithmPoseClustering()
    params = aligner.getDefaults()
    params.setValue(b"max_num_peaks_considered", -1)
    params.setValue(b"pairfinder:distance_MZ:unit", "ppm")
    params.setValue(b"pairfinder:distance_MZ:max_difference", 10.0)
    params.setValue(b"pairfinder:distance_RT:max_difference", 60.0)
    aligner.setParameters(params)
    aligner.setReference(feature_maps[ref_index])

    for feature_map in feature_maps[:ref_index] + feature_maps[ref_index + 1 :]:
        trafo = oms.TransformationDescription()
        aligner.align(feature_map, trafo)
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)


def group_features(feature_maps, output_file):
    oms = _get_oms()
    feature_grouper = oms.FeatureGroupingAlgorithmKD()
    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()
    for i, feature_map in enumerate(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_description.filename = os.path.basename(feature_map.getMetaValue("spectra_data")[0].decode())
        file_description.size = feature_map.size()
        file_descriptions[i] = file_description
    feature_grouper.group(feature_maps, consensus_map)
    consensus_map.setColumnHeaders(file_descriptions)
    consensus_map.setUniqueIds()
    consensus_map.get_df().to_csv(output_file, index=False)


def run_pyopenms(input_dir, output_file, mz_tol, min_fwhm, max_fwhm, noise=1000, sn=5):
    oms = _get_oms()
    input_dir = Path(input_dir).resolve()
    feature_maps = []
    file_count = len(list(input_dir.glob("*.mzML")))

    for file in input_dir.glob("*.mzML"):
        filename = str(file)
        exp = oms.MSExperiment()
        oms.MzMLFile().load(filename, exp)

        mass_traces = []
        mtd = oms.MassTraceDetection()
        mtd_par = mtd.getDefaults()
        mtd_par.setValue(b"mass_error_ppm", mz_tol)
        mtd_par.setValue(b"noise_threshold_int", noise)
        mtd_par.setValue(b"chrom_peak_snr", sn)
        mtd.setParameters(mtd_par)
        mtd.run(exp, mass_traces, 0)

        mass_traces_deconvol = []
        epd = oms.ElutionPeakDetection()
        epd_par = epd.getDefaults()
        epd_par.setValue(b"min_fwhm", min_fwhm)
        epd_par.setValue(b"max_fwhm", max_fwhm)
        epd_par.setValue(b"chrom_peak_snr", sn)
        epd.setParameters(epd_par)
        epd.detectPeaks(mass_traces, mass_traces_deconvol)

        feature_map = oms.FeatureMap()
        ffm = oms.FeatureFindingMetabo()
        ffm_par = ffm.getDefaults()
        ffm_par.setValue(b"local_rt_range", 8.0)
        ffm_par.setValue(b"local_mz_range", 3.5)
        ffm_par.setValue(b"mz_scoring_13C", b"true")
        ffm_par.setValue(b"report_convex_hulls", b"true")
        ffm_par.setValue(b"charge_upper_bound", 2)
        ffm.setParameters(ffm_par)
        ffm.run(mass_traces_deconvol, feature_map, [])

        feature_map.setUniqueIds()
        feature_map.setPrimaryMSRunPath([filename.encode()])
        feature_maps.append(feature_map)

    if file_count == 1:
        feature_maps[0].get_df().to_csv(output_file, index=False)
        return output_file

    align_features(feature_maps)
    group_features(feature_maps, output_file)
    return output_file


def extract_pyopenms_params(config):
    pyopenms_params = config.get("parameters", {}).get("pyopenms", {})
    peak_picking = pyopenms_params.get("peak_picking", {}) if isinstance(pyopenms_params, dict) else {}
    common_params = config.get("common_params", {})
    mz_tol = peak_picking.get("mz_tol", pyopenms_params.get("mz_tol", common_params.get("mz_tolerance_ppm", 10.0)))
    min_fwhm = peak_picking.get("min_fwhm", pyopenms_params.get("min_fwhm", 2.5))
    max_fwhm = peak_picking.get("max_fwhm", pyopenms_params.get("max_fwhm", 60.0))
    noise = peak_picking.get("noise", pyopenms_params.get("noise", 1000))
    sn = peak_picking.get("sn", pyopenms_params.get("sn", 5))
    return {
        "mz_tol": float(mz_tol),
        "min_fwhm": float(min_fwhm),
        "max_fwhm": float(max_fwhm),
        "noise": float(noise),
        "sn": float(sn),
    }


def run_pyopenms_pipeline(config):
    base_dir = get_base_dir()
    input_dir = _resolve_path(base_dir, config["paths"]["input_dir"])
    output_dir = _resolve_path(base_dir, config["paths"]["pyopenms_output"])
    output_file = output_dir / "pyopenms_features.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    params = extract_pyopenms_params(config)
    run_pyopenms(input_dir=input_dir, output_file=output_file, **params)
    if not output_file.exists():
        raise FileNotFoundError(f"pyOpenMS output file not found: {output_file}")
    load_pyopenms_results(output_file.resolve(), input_dir=input_dir, **params)
