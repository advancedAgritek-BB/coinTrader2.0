"""Utility helpers for optional machine learning dependencies."""

ML_AVAILABLE = False

try:  # pragma: no cover - optional dependency detection
    import sklearn  # noqa: F401
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - sklearn not installed
    ML_AVAILABLE = False


def is_ml_available() -> bool:
    """Return True if machine-learning dependencies are available."""
    return ML_AVAILABLE

__all__ = ["is_ml_available", "ML_AVAILABLE"]
