"""Helpers for determining availability of machine learning features."""

# ``ML_AVAILABLE`` indicates whether optional ML dependencies are installed. The
# classifier code is resilient to missing libraries; setting this flag to
# ``True`` allows the caller to attempt ML execution and gracefully handle
# failures via try/except.
ML_AVAILABLE = True

try:  # pragma: no cover - optional dependency
    import lightgbm  # noqa: F401
except Exception:  # pragma: no cover - missing optional deps
    pass
