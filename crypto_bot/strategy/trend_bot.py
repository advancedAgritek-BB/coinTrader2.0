import pandas as pd
from typing import Tuple
import ta


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """Simple trend following strategy using EMA and RSI."""
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    latest = df.iloc[-1]
    score = 0.0
    direction = 'none'

    if latest['ema20'] > latest['ema50'] and latest['rsi'] > 55:
        score = min((latest['rsi'] - 50) / 50, 1)
        direction = 'long'
    elif latest['ema20'] < latest['ema50'] and latest['rsi'] < 45:
        score = min((50 - latest['rsi']) / 50, 1)
        direction = 'short'

    return score, direction
