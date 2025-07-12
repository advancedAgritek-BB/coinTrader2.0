"""Simple cross-exchange arbitrage signal."""

from typing import Optional, Tuple

import pandas as pd


def generate_signal(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Detect price discrepancies between two exchanges.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        DataFrames containing ``close`` price columns for each exchange.
    config : dict, optional
        Optional configuration with ``spread_threshold`` key.

    Returns
    -------
    tuple[float, str]
        Normalized score and trade direction. ``"long"`` means buy on ``a`` and
        sell on ``b``. ``"short"`` means the opposite.
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return 0.0, "none"

    threshold = 0.002
    if config:
        threshold = float(config.get("spread_threshold", threshold))

    price_a = float(df_a["close"].iloc[-1])
    price_b = float(df_b["close"].iloc[-1])
    if price_b == 0:
        return 0.0, "none"

    spread = (price_a - price_b) / price_b
    score = abs(spread)

    if spread > threshold:
        return min(score, 1.0), "short"
    if spread < -threshold:
        return min(score, 1.0), "long"
    return 0.0, "none"
