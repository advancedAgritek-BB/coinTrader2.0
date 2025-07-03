import pandas as pd
import numpy as np
import ta


def classify_regime(df: pd.DataFrame) -> str:
    """Classify market regime. Requires at least 20 rows of data."""

    if df is None or df.empty or len(df) < 20:
        return "unknown"

    df = df.copy()

    df['ema20'] = (
        ta.trend.ema_indicator(df['close'], window=20)
        if len(df) >= 20
        else np.nan
    )
    df['ema50'] = (
        ta.trend.ema_indicator(df['close'], window=50)
        if len(df) >= 50
        else np.nan
    )

    if len(df) >= 14:
        try:
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14
            )
        except IndexError:
            df['adx'] = np.nan
            df['rsi'] = np.nan
            df['atr'] = np.nan
            return "unknown"
    else:
        df['adx'] = np.nan
        df['rsi'] = np.nan
        df['atr'] = np.nan

    if len(df) >= 20:
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_width'] = bb.bollinger_wband()
    else:
        df['bb_width'] = np.nan

    volume_ma20 = df['volume'].rolling(20).mean() if len(df) >= 20 else pd.Series(np.nan, index=df.index)
    atr_ma20 = df['atr'].rolling(20).mean() if len(df) >= 20 else pd.Series(np.nan, index=df.index)

    latest = df.iloc[-1]
    regime = 'sideways'

    # Trending
    if latest['adx'] > 25 and latest['ema20'] > latest['ema50']:
        regime = 'trending'
    # Sideways
    elif latest['adx'] < 20 and latest['bb_width'] < 5:
        regime = 'sideways'
    # Breakout
    elif (
        latest['bb_width'] < 4
        and not np.isnan(volume_ma20.iloc[-1])
        and latest['volume'] > volume_ma20.iloc[-1] * 2
    ):
        regime = 'breakout'
    # Mean-reverting
    elif 30 <= latest['rsi'] <= 70 and abs(latest['close'] - latest['ema20']) / latest['close'] < 0.01:
        regime = 'mean-reverting'
    # Volatile
    elif (
        not np.isnan(atr_ma20.iloc[-1])
        and latest['atr'] > atr_ma20.iloc[-1] * 1.5
    ):
        regime = 'volatile'

    return regime
