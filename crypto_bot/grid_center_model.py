"""Predict grid centre price placeholder.

This module exposes :func:`predict_centre` which currently returns
``float('nan')``. Real predictions may be plugged in later if a
machine learning model becomes available.
"""

from __future__ import annotations

import pandas as pd


def predict_centre(df: pd.DataFrame) -> float:
    """Return predicted centre price for a grid strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Recent OHLCV data.

    Returns
    -------
    float
        Placeholder value ``float('nan')``. Replace with a real model
        prediction when one is implemented.
    """
    return float("nan")
