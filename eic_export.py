"""Compatibility shim.

EIC export has been unified into lipidbench.main (CLI + YAML).
Use: python main.py --export-eic ...
"""

from lipidbench.main import main


if __name__ == "__main__":
    raise SystemExit(main())
