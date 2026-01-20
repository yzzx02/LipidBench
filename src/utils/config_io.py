import yaml
from pathlib import Path
import os
Base_dir = Path(__file__).resolve().parent.parent.parent
Config_path = os.path.join(Base_dir,'config.yaml')
def load_config():
    if not os.path.exists(Config_path):
        raise FileNotFoundError(f"Configuration file not found at {Config_path}")
    with open(Config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def get_base_dir():
    return Base_dir