from utils.data_io import load_msdial_results
from utils.config_io import get_base_dir, _resolve_path
from pathlib import Path
def run_msdial_pipeline(config):
    msdial_params = config.get('parameters',{}).get('msdial',{})
    input_dir = msdial_params.get('input_dir','')
    output_dir = msdial_params.get('output_dir','')
    if not input_dir:
        raise ValueError("Input directory must be specified in the configuration.")
    input_dir = _resolve_path(get_base_dir(),input_dir)
    #确保输出目录存在
    output_dir = _resolve_path(get_base_dir(),output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    #读取所有xlsx文件一个一个处理
    for file in input_dir.glob("*.xlsx"):
        file_path = Path(file).resolve()
        # Load and process MS-DIAL results
        output_file = output_dir / f"{file.stem}_processed.csv"
        load_msdial_results(file_path, output_file)
        # Save processed data