from pathlib import Path
import os


def _resolve_path(base_dir, maybe_relative_path):
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'PyYAML'. Install it with: pip install pyyaml"
        ) from e

    if not os.path.exists(Config_path):
        raise FileNotFoundError(f"Configuration file not found at {Config_path}")
    with open(Config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def get_base_dir():
    return Base_dir


Base_dir = Path(__file__).resolve().parent.parent.parent
Config_path = _resolve_path(Base_dir, "config.yaml")
