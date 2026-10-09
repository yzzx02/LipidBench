from pathlib import Path

from lipidbench.utils.config_io import load_config
from lipidbench.utils.local_paths import expand_local_paths, get_cache_root, get_local_root


def test_default_local_storage_is_outside_checkout(monkeypatch):
    monkeypatch.delenv("CHROMAPEAK_LOCAL_ROOT", raising=False)
    monkeypatch.delenv("CHROMAPEAK_CACHE_ROOT", raising=False)
    root = Path(__file__).resolve().parents[1]
    assert get_local_root() == root.parent / (root.name + "-local")
    assert not get_cache_root().is_relative_to(root)
    config = load_config()
    assert not Path(config["paths"]["input_dir"]).is_relative_to(root)
    assert not Path(config["paths"]["output_dir"]).is_relative_to(root)


def test_local_paths_honor_overrides_without_rewriting_explicit_paths(monkeypatch, tmp_path):
    local = tmp_path / "local storage"
    cache = tmp_path / "runtime cache"
    monkeypatch.setenv("CHROMAPEAK_LOCAL_ROOT", str(local))
    monkeypatch.setenv("CHROMAPEAK_CACHE_ROOT", str(cache))
    original = {"paths": ["${CHROMAPEAK_LOCAL_ROOT}/inputs", "${CHROMAPEAK_CACHE_ROOT}/locks"],
                "explicit": "/chosen/input", "parameter": 10.0}
    result = expand_local_paths(original)
    assert result["paths"] == [str(local) + "/inputs", str(cache) + "/locks"]
    assert result["explicit"] == original["explicit"]
    assert result["parameter"] == original["parameter"]
    assert "${CHROMAPEAK_LOCAL_ROOT}" in original["paths"][0]
