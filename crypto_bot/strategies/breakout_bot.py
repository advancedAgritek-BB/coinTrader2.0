import pandas as pd
import ta
from typing import Any, Dict


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    *,
    config: Dict[str, Any] | None = None,
) -> tuple[float, str, Dict[str, float]]:
    """Return breakout signal using rolling max distance as intensity.

    Parameters
    ----------
    df:
        OHLCV dataframe.
    symbol:
        Trading pair symbol. Unused but kept for signature compatibility.
    timeframe:
        Candle timeframe string (e.g. ``"1h"``).
    config:
        Optional configuration mapping. Expected structure::

            {
                "breakout": {
                    "donchian_len": int,
                    "bb_length": int,
                    "bb_std": float,
                    "kc_length": int,
                    "kc_mult": float,
                    "volume_window": int,
                    "vol_multiplier": float,
                    "min_intensity": {"1h": 0.015, ...}
                }
            }

    Returns
    -------
    tuple
        ``(metric, signal, meta)`` where ``metric`` is the breakout intensity
        and ``signal`` is ``"long"`` or ``"none"``.
    """

    cfg = config or {}
    brk = cfg.get("breakout", {})
    N = int(brk.get("donchian_len", brk.get("dc_length", 20)))
    bb_len = int(brk.get("bb_length", 20))
    bb_std = float(brk.get("bb_std", 2.0))
    kc_len = int(brk.get("kc_length", 20))
    kc_mult = float(brk.get("kc_mult", 1.5))
    vol_window = int(brk.get("volume_window", 20))
    vol_multiplier = float(brk.get("vol_multiplier", 1.0))
    min_intensity = float(brk.get("min_intensity", {}).get(timeframe or "", 0.0))

    lookback = max(N, bb_len, kc_len, vol_window) + 1
    if len(df) < lookback:
        return 0.0, "none", {"reason": "insufficient_data"}

    recent = df.tail(lookback)
    close = recent["close"]
    high = recent["high"]
    low = recent["low"]
    volume = recent.get("volume", pd.Series([], dtype=float))

    bb = ta.volatility.BollingerBands(close, window=bb_len, window_dev=bb_std)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    atr = ta.volatility.average_true_range(high, low, close, window=kc_len)
    kc_width = 2 * atr * kc_mult
    squeeze = bool(bb_width.iloc[-1] < kc_width.iloc[-1])

    vol_ma = volume.rolling(vol_window).mean()
    vol_ok = bool(
        volume.iloc[-1] > vol_ma.iloc[-1] * vol_multiplier if vol_ma.iloc[-1] > 0 else False
    )

    rolling_max = high.iloc[-N:].max()
    metric = (close.iloc[-1] - rolling_max) / rolling_max if rolling_max else 0.0

    if squeeze and vol_ok and metric >= min_intensity:
        return metric, "long", {"rolling_max": float(rolling_max)}
    return metric, "none", {}
