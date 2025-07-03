from typing import Optional, Tuple
import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
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

    if score > 0 and (config is None or config.get('atr_normalization', True)):
        score = normalize_score_by_volatility(df, score)

    return score, direction
