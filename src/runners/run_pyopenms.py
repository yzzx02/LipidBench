import pyopenms as oms
import os
from pathlib import Path
from utils.config_io import get_base_dir, _resolve_path
def align_features(feature_maps):
    ref_index = feature_maps.index(sorted(feature_maps, key=lambda x: x.size())[-1])
    aligner = oms.MapAlignmentAlgorithmPoseClustering()
    trafos = {}
    #parameter setting
    params = aligner.getDefaults()
    params.setValue(b'max_num_peaks_considered', -1)
    params.setValue(b"pairfinder:distance_MZ:unit", "ppm")    
    params.setValue(b'pairfinder:distance_MZ:max_difference', 10.0)
    params.setValue(b'pairfinder:distance_RT:max_difference', 60.0)
    aligner.setParameters(params)
    aligner.setReference(feature_maps[ref_index])
    #align each feature map to the reference
    for feature_map in feature_maps[:ref_index] + feature_maps[ref_index + 1 :]:
        trafo = oms.TransformationDescription()  # save the transformed data points
        aligner.align(feature_map, trafo)
        trafos[feature_map.getMetaValue("spectra_data")[0].decode()] = trafo
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)
def group_features(feature_maps, output_file):
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
    df = consensus_map.get_df()
    df.to_csv(output_file,index=False)
def run_pyopenms(input_dir,output_file,mz_tol,min_fwhm,max_fwhm,noise=1000,sn=5):
    input_dir = Path(input_dir).resolve()
    feature_maps = []
    #判断文件数目
    file_count = len(list(input_dir.glob("*.mzML")))
    #如果只有一个文件则不对齐直接输出feature
    for file in input_dir.glob("*.mzML"):
        filename = str(file)
        exp = oms.MSExperiment()
        oms.MzMLFile().load(filename, exp)
        # Perform mass trace detection
        mass_traces =[]
        mtd=oms.MassTraceDetection()
        mtd_par=mtd.getDefaults()
        mtd_par.setValue(b"mass_error_ppm", mz_tol)
        mtd_par.setValue(b'noise_threshold_int', noise)
        mtd_par.setValue(b'chrom_peak_snr', sn)
        mtd.setParameters(mtd_par)
        mtd.run(exp, mass_traces, 0)
        #  elution peak detection
        mass_traces_deconvol = []
        epd = oms.ElutionPeakDetection()
        epd_par = epd.getDefaults()
        epd_par.setValue(b'min_fwhm', min_fwhm)
        epd_par.setValue(b'max_fwhm', max_fwhm)
        epd_par.setValue(b'chrom_peak_snr', sn)
        epd.setParameters(epd_par)
        epd.detectPeaks(mass_traces, mass_traces_deconvol)
        # feature detection
        feature_map=oms.FeatureMap()
        ffm=oms.FeatureFindingMetabo()
        ffm_par=ffm.getDefaults()
        ffm_par.setValue(b'local_rt_range', 8.0)
        ffm_par.setValue(b'local_mz_range', 3.5)
        ffm_par.setValue(b'mz_scoring_13C', b'true')
        ffm_par.setValue(b'charge_upper_bound',2)
        ffm.setParameters(ffm_par)
        ffm.run(mass_traces_deconvol, feature_map,[])
        # Add metadata
        feature_map.setUniqueIDs()
        feature_map.setPrimaryMSRunPath([file.encode()])
         #直接输出feature_maps
        feature_maps.append(feature_map)
        if file_count==1:
            return feature_map.get_df().to_csv(output_file,index=False)
        else:
            #aline features across samples
            align_features(feature_maps)
            group_features(feature_maps, output_file)

def extract_pyopenms_params(config):
    pyopenms_params = config.get("parameters", {}).get("pyopenms", {})
    common_params = config.get("common_params", {})
    mz_tol = pyopenms_params.get("mz_tol", common_params.get("mz_tolerance_ppm", 10.0))
    min_fwhm = pyopenms_params.get("min_fwhm", 2.5)
    max_fwhm = pyopenms_params.get("max_fwhm", 60.0)
    noise = pyopenms_params.get('noise',1000)
    sn = pyopenms_params.get('sn',5)
    return {
        'mz_tol': float(mz_tol),
        'min_fwhm': float(min_fwhm),
        'max_fwhm': float(max_fwhm),
        'noise': float(noise),
        'sn': float(sn),
    }


def run_pyopenms_pipeline(config):
    base_dir = get_base_dir()
    input_dir = _resolve_path(base_dir, config['paths']['input_dir'])
    output_dir = _resolve_path(base_dir, config['paths']['output_dir'])
    output_file = os.path.join(output_dir, "pyopenms_features.csv")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    params = extract_pyopenms_params(config)
    df=run_pyopenms(input_dir=input_dir, output_file=output_file, **params)
    # 删除无关列：sequence(序列), charge(电荷), quality(质量评分)
    cols_to_drop = ['sequence', 'charge', 'quality']
    df = df.drop(columns=[c for c in df.columns if c in cols_to_drop], errors='ignore')

    # 对RT列除以60,3位小数，mz列保留4位小数
    if 'mz' in df.columns:
        df['mz'] = df['mz'].round(4)
    if 'RT' in df.columns:
        df['RT'] = (df['RT'] / 60.0).round(3)

    if 'Feature_id' not in df.columns:
        df.insert(0, 'Feature_id', range(1, len(df) + 1))
    
    # 保存 CSV，float_format='%.4f' 可以避免科学计数法
    df.to_csv(output_file, index=False, float_format='%.4f')

if __name__=='__main__':
    feature_grouper = oms.FeatureGroupingAlgorithmKD()
    print(feature_grouper.getDefaults())

