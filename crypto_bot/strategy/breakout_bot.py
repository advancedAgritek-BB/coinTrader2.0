from typing import Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Breakout strategy with volatility contraction and momentum checks.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data ordered oldest -> newest.
    config : dict, optional
        Optional configuration overriding the default thresholds:
        ``contraction_threshold`` (float), ``consolidation_period`` (int),
        ``volume_multiple`` (float) and ``slope_window`` (int).
    """

    contraction_threshold = 0.9
    consolidation_period = 5
    volume_multiple = 2.0
    slope_window = 5

    if config:
        contraction_threshold = config.get("contraction_threshold", contraction_threshold)
        consolidation_period = config.get("consolidation_period", consolidation_period)
        volume_multiple = config.get("volume_multiple", volume_multiple)
        slope_window = config.get("slope_window", slope_window)

    if len(df) < max(40, consolidation_period + 20):
        return 0.0, "none"

    close = df["close"]
    volume = df["volume"]

    macd = ta.trend.macd_diff(close)

    bb = ta.volatility.BollingerBands(close)
    hband = bb.bollinger_hband()
    lband = bb.bollinger_lband()
    width = hband - lband
    width_mean = width.rolling(20).mean()

    contraction = width < width_mean * contraction_threshold
    if not contraction.iloc[-consolidation_period:-1].all():
        return 0.0, "none"

    vol_mean = volume.rolling(20).mean()
    vol_spike = volume.iloc[-1] > vol_mean.iloc[-1] * volume_multiple

    ema_fast = ta.trend.ema_indicator(close, window=slope_window)
    ema_slope = ema_fast.iloc[-1] - ema_fast.iloc[-2]

    direction: str
    if close.iloc[-1] > hband.iloc[-1] and macd.iloc[-1] > 0:
        momentum_ok = vol_spike or ema_slope > 0
        if not momentum_ok:
            return 0.0, "none"
        score, direction = 1.0, "long"
    elif close.iloc[-1] < lband.iloc[-1] and macd.iloc[-1] < 0:
        momentum_ok = vol_spike or ema_slope < 0
        if not momentum_ok:
            return 0.0, "none"
        score, direction = 1.0, "short"
    else:
        return 0.0, "none"

    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    return score, direction
