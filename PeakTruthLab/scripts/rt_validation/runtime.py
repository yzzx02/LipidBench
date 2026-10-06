"""Path and device configuration for the archived RT experiment replay."""
from pathlib import Path
import json
import os
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('CHROMAPEAK_PROJECT_ROOT') or SCRIPT_DIR.parents[2]).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
WORK = Path(os.environ.get('CHROMAPEAK_RT_WORKDIR') or PROJECT_ROOT / 'work/rt_validation').resolve()
CONFIG_PATH = WORK / 'runtime_config.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf8')) if CONFIG_PATH.exists() else {}
OUTPUT = WORK / 'results'
CHECKPOINT = Path(CONFIG.get('checkpoint') or WORK / 'best_detection.pt')
RSCRIPT = Path(CONFIG.get('rscript') or shutil.which('Rscript') or 'Rscript')


def resolve_device(name='auto'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Install a matching ROCm/CUDA torch + torchvision pair, or explicitly use --device cpu.')
    # ROCm intentionally exposes its GPU through the torch.cuda API.
    return torch.device(name)
