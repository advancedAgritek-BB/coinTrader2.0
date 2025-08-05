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

        model_path = Path(__file__).resolve().parent.parent / "models" / "meta_selector_lgbm.txt"
        with open(model_path, "r", encoding="utf-8") as file:
            file.read(1)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", exc)
        return False


ML_AVAILABLE: bool = is_ml_available()

__all__ = ["is_ml_available", "ML_AVAILABLE"]

