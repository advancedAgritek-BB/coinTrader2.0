import pandas as pd
import ta


def generate_signal(
    df: pd.DataFrame,
    config: dict | None = None,
    *,
    pyth_price: float | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> tuple[float, str]:
    """Return a simple Solana scalping signal using RSI and MACD.

    Parameters
    ----------
    df : pd.DataFrame
        Minute level OHLCV data.
    config : dict, optional
        Optional configuration overriding default indicator windows.
    """
    if df is None or df.empty:
        return 0.0, "none"

    if pyth_price is not None and not df.empty:
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("close")] = float(pyth_price)

    params = config.get("solana_scalping", {}) if config else {}
    rsi_window = int(params.get("rsi_window", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))
    macd_fast = int(params.get("macd_fast", 12))
    macd_slow = int(params.get("macd_slow", 26))
    macd_signal = int(params.get("macd_signal", 9))

    lookback = max(rsi_window, macd_slow, macd_signal)
    if len(df) < lookback:
        return 0.0, "none"

    rsi = ta.momentum.rsi(df["close"], window=rsi_window)
    macd_hist = ta.trend.macd_diff(
        df["close"],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_signal,
    )

    rsi_val = float(rsi.iloc[-1])
    hist_val = float(macd_hist.iloc[-1])

    if pd.isna(rsi_val) or pd.isna(hist_val):
        return 0.0, "none"

    score = min(abs(hist_val) / df["close"].iloc[-1], 1.0)
    if rsi_val < oversold and hist_val > 0:
        return score, "long"
    if rsi_val > overbought and hist_val < 0:
        return score, "short"

    return 0.0, "none"
