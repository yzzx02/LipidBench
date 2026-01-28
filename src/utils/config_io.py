import yaml
from pathlib import Path
import os
    #路径自动拼接
def _resolve_path(base_dir, maybe_relative_path):
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
def load_config():
    if not os.path.exists(Config_path):
        raise FileNotFoundError(f"Configuration file not found at {Config_path}")
    with open(Config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def get_base_dir():
    return Base_dir
Base_dir = Path(__file__).resolve().parent.parent.parent
Config_path = _resolve_path(Base_dir,'config.yaml')

