import pandas as pd
from typing import Tuple
import ta


def generate_signal(df: pd.DataFrame) -> Tuple[float, str]:
    """Breakout strategy using MACD and volume."""
    macd = ta.trend.macd_diff(df['close'])
    vol_mean = df['volume'].rolling(20).mean()
    if macd.iloc[-1] > 0 and df['volume'].iloc[-1] > vol_mean.iloc[-1] * 2:
        return 1.0, 'long'
    elif macd.iloc[-1] < 0 and df['volume'].iloc[-1] > vol_mean.iloc[-1] * 2:
        return 1.0, 'short'
    return 0.0, 'none'
