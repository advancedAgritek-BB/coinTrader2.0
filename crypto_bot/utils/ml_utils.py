"""Utilities for optional machine-learning dependencies.

The module tries to import optional ML libraries (e.g., LightGBM) and
exposes :data:`ML_AVAILABLE` indicating whether these features are available.
"""

from __future__ import annotations

try:  # pragma: no cover - runtime check
    import lightgbm  # type: ignore  # noqa: F401

    ML_AVAILABLE = True
except Exception:  # pragma: no cover - broad to cover missing deps
    ML_AVAILABLE = False
