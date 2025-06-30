import pandas as pd
from typing import Tuple
import ta


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """Mean reversion using RSI."""
    rsi = ta.momentum.rsi(df['close'], window=14)
    latest = rsi.iloc[-1]
    if latest < 30:
        score = min((30 - latest) / 30, 1)
        return score, 'long'
    elif latest > 70:
        score = min((latest - 70) / 30, 1)
        return score, 'short'
    return 0.0, 'none'
