"""Legacy entrypoint kept for compatibility; prefer running project-root main.py."""

from pathlib import Path
import sys

ROOT_MAIN = Path(__file__).resolve().parent.parent / "main.py"
if ROOT_MAIN.exists():
    sys.path.insert(0, str(ROOT_MAIN.parent / "src"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("project_root_main", ROOT_MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    module.main()
else:
    raise RuntimeError("Root-level main.py not found; please run from project root.")
