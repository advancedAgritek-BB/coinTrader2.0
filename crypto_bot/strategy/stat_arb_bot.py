from __future__ import annotations

"""Simple statistical arbitrage strategy using spread z-score."""

from typing import Optional, Tuple

import pandas as pd
from crypto_bot.utils import stats


_ZSCORE_THRESHOLD_DEFAULT = 2.0
_LOOKBACK_DEFAULT = 20
_CORRELATION_THRESHOLD = 0.8


def generate_signal(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return score and direction based on the spread z-score.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        DataFrames containing at least a ``close`` column.
    config : dict, optional
        Configuration with ``zscore_threshold`` and ``lookback``.
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return 0.0, "none"

    threshold = float(config.get("zscore_threshold", _ZSCORE_THRESHOLD_DEFAULT)) if config else _ZSCORE_THRESHOLD_DEFAULT
    lookback = int(config.get("lookback", _LOOKBACK_DEFAULT)) if config else _LOOKBACK_DEFAULT

    if len(df_a) < lookback or len(df_b) < lookback:
        return 0.0, "none"

    corr = df_a["close"].corr(df_b["close"])
    if pd.isna(corr) or corr < _CORRELATION_THRESHOLD:
        return 0.0, "none"

    spread = df_a["close"] - df_b["close"]
    z = stats.zscore(spread, lookback)
    if z.empty:
        return 0.0, "none"

    z_last = z.iloc[-1]
    if abs(z_last) <= threshold:
        return 0.0, "none"

    direction = "long" if z_last < 0 else "short"
    score = float(abs(z_last))
    return score, direction
