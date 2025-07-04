import pandas as pd
import ta
from typing import Optional, Tuple

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return signal for quick bounce trades using RSI extremes."""
    if df.empty:
        return 0.0, "none"

    params = config.get("bounce_scalper", {}) if config else {}
    window = int(params.get("rsi_window", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))

    if len(df) < window:
        return 0.0, "none"

    rsi_series = ta.momentum.rsi(df["close"], window=window)
    rsi = rsi_series.iloc[-1]
    if pd.isna(rsi):
        return 0.0, "none"

    if rsi < oversold:
        score = min((oversold - rsi) / oversold, 1.0)
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    if rsi > overbought:
        score = min((rsi - overbought) / (100 - overbought), 1.0)
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"
    return 0.0, "none"
