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
