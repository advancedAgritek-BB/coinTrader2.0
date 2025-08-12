"""Utility helpers for optional machine learning dependencies."""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)

_REQUIRED_PACKAGES: Iterable[str] = ("sklearn", "joblib", "ta")

_LOGGER_ONCE = {"ml_unavailable": False}


def warn_ml_unavailable_once() -> None:
    """Log a one-time notice when ML components are missing."""
    if not _LOGGER_ONCE["ml_unavailable"]:
        logger.info(
            "Machine learning model not found; running without ML features."
        )
        _LOGGER_ONCE["ml_unavailable"] = True


def _check_packages(pkgs: Iterable[str]) -> bool:
    """Return ``True`` if all packages in ``pkgs`` can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in pkgs)


def is_ml_available() -> bool:
    """Return ``True`` if optional ML dependencies and model are available."""
    try:
        if not _check_packages(_REQUIRED_PACKAGES):
            raise ImportError("Missing required ML packages")

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Missing Supabase credentials")

        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "meta_selector_lgbm.txt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", exc)
        return False


ML_AVAILABLE = is_ml_available()

__all__ = ["is_ml_available", "ML_AVAILABLE", "warn_ml_unavailable_once"]

