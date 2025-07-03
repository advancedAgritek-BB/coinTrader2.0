from typing import Optional, Tuple
import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Mean reversion using RSI."""
    rsi = ta.momentum.rsi(df['close'], window=14)
    latest = rsi.iloc[-1]
    if latest < 30:
        score = min((30 - latest) / 30, 1)
        direction = 'long'
    elif latest > 70:
        score = min((latest - 70) / 30, 1)
        direction = 'short'
    else:
        return 0.0, 'none'

    if config is None or config.get('atr_normalization', True):
        score = normalize_score_by_volatility(df, score)
    return score, direction
