"""Deprecated module.

EIC export pipeline has been unified to:
  lipidbench.eic.extract_eic_pyopenms.build()

Please update callers to use the unified build-based flow.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise AttributeError(
        "lipidbench.eic.export is deprecated. "
        "Use lipidbench.eic.extract_eic_pyopenms.build instead."
    )
