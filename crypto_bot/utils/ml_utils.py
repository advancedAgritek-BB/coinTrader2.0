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
"""Utility functions for optional machine learning support."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def is_ml_available() -> bool:
    """Return ``True`` if optional ML dependencies and model are available."""
    try:
        # Minimal imports to verify heavy optional dependencies
        import numpy  # type: ignore  # noqa: F401
        import pandas  # type: ignore  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # type: ignore  # noqa: F401

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Missing Supabase credentials")

        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "meta_selector_lgbm.txt"
        )
        with open(model_path, "r", encoding="utf-8") as f:
            f.read(1)
        return True
    except ImportError as e:
        logger.error("ML unavailable: %s", e)
    except ValueError as e:
        logger.error("ML unavailable: %s", e)
    except Exception as e:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", e)
    return False


ML_AVAILABLE = is_ml_available()
"""Utilities for optional machine learning components."""
from __future__ import annotations

import importlib.util
from typing import Iterable

__all__=["ML_AVAILABLE"]

# Required packages for ML features to be considered available
_REQUIRED_PACKAGES: Iterable[str]=("sklearn","joblib","ta")

def _check_packages(pkgs: Iterable[str]) -> bool:
    """Return True if all packages in ``pkgs`` can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in pkgs)

# Indicates whether optional ML dependencies are installed and usable
ML_AVAILABLE: bool=_check_packages(_REQUIRED_PACKAGES)
