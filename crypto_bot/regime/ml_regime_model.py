from __future__ import annotations

import pandas as pd


def predict_regime(df: pd.DataFrame) -> str:
    """Return a simple regime prediction using price momentum.

    The implementation is intentionally lightweight and serves as a
    placeholder for a trained machine learning model.  When the
    cumulative return over the provided data is positive ``"trending"``
    is returned, otherwise ``"sideways"`` is assumed.
    """
    if df is None or len(df) < 2:
        return "unknown"

    change = df["close"].iloc[-1] - df["close"].iloc[0]
    return "trending" if change > 0 else "sideways"
