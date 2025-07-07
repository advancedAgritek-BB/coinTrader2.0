from typing import Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility


def _squeeze(
    df: pd.DataFrame,
    bb_len: int,
    bb_std: float,
    kc_len: int,
    kc_mult: float,
    threshold: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return squeeze boolean series and ATR values."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb = ta.volatility.BollingerBands(close, window=bb_len, window_dev=bb_std)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    atr = ta.volatility.average_true_range(high, low, close, window=kc_len)
    kc_width = 2 * atr * kc_mult

    squeeze = (bb_width / bb_mid < threshold) & (bb_width < kc_width)
    return squeeze, atr


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    higher_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, str, float]:
    """Breakout strategy using Bollinger/Keltner squeeze confirmation.

    Returns
    -------
    Tuple[float, str, float]
        Signal score, direction and ATR value used for stop distance.
    """
    if df is None or df.empty:
        return 0.0, "none", 0.0

    cfg = (config or {}).get("breakout", {})
    bb_len = int(cfg.get("bb_length", 20))
    bb_std = float(cfg.get("bb_std", 2))
    kc_len = int(cfg.get("kc_length", 20))
    kc_mult = float(cfg.get("kc_mult", 1.5))
    dc_len = int(cfg.get("dc_length", 20))
    atr_buffer_mult = float(cfg.get("atr_buffer_mult", 0.1))
    vol_window = int(cfg.get("volume_window", 20))
    volume_mult = float(cfg.get("volume_mult", 2))
    threshold = float(cfg.get("squeeze_threshold", 0.03))
    momentum_filter = bool(cfg.get("momentum_filter", False))

    lookback = max(bb_len, kc_len, dc_len, vol_window, 14)
    if len(df) < lookback:
        return 0.0, "none", 0.0

    squeeze, atr = _squeeze(df, bb_len, bb_std, kc_len, kc_mult, threshold)
    if pd.isna(squeeze.iloc[-1]) or not squeeze.iloc[-1]:
        return 0.0, "none", 0.0

    if higher_df is not None and not higher_df.empty:
        h_sq, _ = _squeeze(higher_df, bb_len, bb_std, kc_len, kc_mult, threshold)
        if pd.isna(h_sq.iloc[-1]) or not h_sq.iloc[-1]:
            return 0.0, "none", 0.0

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    dc_high = high.rolling(dc_len).max().shift(1)
    dc_low = low.rolling(dc_len).min().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    rsi = ta.momentum.rsi(close, window=14)
    macd_hist = ta.trend.macd_diff(close)

    vol_ok = volume.iloc[-1] > vol_ma.iloc[-1] * volume_mult if vol_ma.iloc[-1] > 0 else False
    atr_last = atr.iloc[-1]
    upper_break = dc_high.iloc[-1] + atr_last * atr_buffer_mult
    lower_break = dc_low.iloc[-1] - atr_last * atr_buffer_mult

    long_cond = close.iloc[-1] > upper_break
    short_cond = close.iloc[-1] < lower_break

    if momentum_filter:
        long_cond = long_cond and (rsi.iloc[-1] > 50 or macd_hist.iloc[-1] > 0)
        short_cond = short_cond and (rsi.iloc[-1] < 50 or macd_hist.iloc[-1] < 0)

    direction = "none"
    score = 0.0
    if long_cond and vol_ok:
        direction = "long"
        score = 1.0
    elif short_cond and vol_ok:
        direction = "short"
        score = 1.0

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(df, score)

    if higher_df is not None:
        return score, direction
    return score, direction, atr_last
