import pandas as pd
import ta


def classify_regime(df: pd.DataFrame) -> str:
    """Classify market regime based on technical indicators."""
    if len(df) < 14:
        return "unknown"
    df = df.copy()
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'] = bb.bollinger_wband()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    latest = df.iloc[-1]
    regime = 'sideways'

    # Trending
    if latest['adx'] > 25 and latest['ema20'] > latest['ema50']:
        regime = 'trending'
    # Sideways
    elif latest['adx'] < 20 and latest['bb_width'] < 5:
        regime = 'sideways'
    # Breakout
    elif latest['bb_width'] < 4 and latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 2:
        regime = 'breakout'
    # Mean-reverting
    elif 30 <= latest['rsi'] <= 70 and abs(latest['close'] - latest['ema20']) / latest['close'] < 0.01:
        regime = 'mean-reverting'
    # Volatile
    elif latest['atr'] > df['atr'].rolling(20).mean().iloc[-1] * 1.5:
        regime = 'volatile'

    return regime
