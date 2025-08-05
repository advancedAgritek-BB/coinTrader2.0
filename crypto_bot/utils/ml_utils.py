from __future__ import annotations

"""Helpers for optional machine-learning dependencies.

The project can leverage packages like LightGBM or scikit-learn when they are
installed. Import errors are ignored so the rest of the system can operate
without these heavy dependencies.
"""

try:  # pragma: no cover - optional dependency detection
    import lightgbm  # type: ignore  # noqa: F401
    import sklearn  # type: ignore  # noqa: F401
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - missing optional deps
    ML_AVAILABLE = False


def is_ml_available() -> bool:
    """Return ``True`` if optional ML libraries are available."""
    return ML_AVAILABLE


__all__ = ["ML_AVAILABLE", "is_ml_available"]
