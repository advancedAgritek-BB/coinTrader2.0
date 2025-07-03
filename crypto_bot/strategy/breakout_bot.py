from typing import Optional, Tuple
import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Breakout strategy using MACD and volume."""
    macd = ta.trend.macd_diff(df['close'])
    vol_mean = df['volume'].rolling(20).mean()
    if macd.iloc[-1] > 0 and df['volume'].iloc[-1] > vol_mean.iloc[-1] * 2:
        score, direction = 1.0, 'long'
    elif macd.iloc[-1] < 0 and df['volume'].iloc[-1] > vol_mean.iloc[-1] * 2:
        score, direction = 1.0, 'short'
    else:
        return 0.0, 'none'

    if config is None or config.get('atr_normalization', True):
        score = normalize_score_by_volatility(df, score)
    return score, direction
