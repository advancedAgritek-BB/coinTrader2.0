"""Utilities for optional machine learning dependencies."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)

# Required packages for ML features to be considered available
_REQUIRED_PACKAGES: Iterable[str] = ("sklearn", "joblib", "ta")


def _check_packages(pkgs: Iterable[str]) -> bool:
    """Return ``True`` if all packages in ``pkgs`` can be imported."""
"""Utility helpers for optional machine learning dependencies."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable
import importlib

    return all(importlib.util.find_spec(name) is not None for name in pkgs)

# Required packages for ML features to be considered available
_REQUIRED_PACKAGES: Iterable[str] = (
    "sklearn",
    "joblib",
    "ta",
)


def _check_packages(pkgs: Iterable[str]) -> bool:
    """Return ``True`` if all packages in ``pkgs`` can be imported."""
    for name in pkgs:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


def is_ml_available() -> bool:
    """Return ``True`` if optional ML dependencies and model are available."""

    try:
        if not _check_packages(_REQUIRED_PACKAGES):
            raise ImportError("Missing required ML packages")
    if not _check_packages(_REQUIRED_PACKAGES):
        return False

    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("Missing Supabase credentials")

        model_path = Path(__file__).resolve().parent.parent / "models" / "meta_selector_lgbm.txt"
        with open(model_path, "r", encoding="utf-8") as file:
            file.read(1)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", exc)
        return False


ML_AVAILABLE: bool = is_ml_available()

__all__ = ["is_ml_available", "ML_AVAILABLE"]

    except Exception as e:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", e)
        return False


# Indicates whether optional ML dependencies are installed and usable
ML_AVAILABLE: bool = is_ml_available()

__all__ = ["is_ml_available", "ML_AVAILABLE"]

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
