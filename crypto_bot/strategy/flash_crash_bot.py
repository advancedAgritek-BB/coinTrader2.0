from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility

NAME = "flash_crash_bot"

def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return long signal on sudden drops with high volume."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")
    if df is None or len(df) < 2:
        return 0.0, "none"

    params = config.get("flash_crash", {}) if config else {}
    drop_pct = float(params.get("drop_pct", 0.1))
    vol_mult = float(params.get("volume_mult", 5.0))
    vol_window = int(params.get("vol_window", 20))
    ema_window = int(params.get("ema_window", 200))
    atr_norm = bool(params.get("atr_normalization", True))

    lookback = max(vol_window, ema_window)
    recent = df.iloc[-(lookback + 1) :]

    vol_ma = recent["volume"].rolling(vol_window).mean()
    ema = ta.trend.ema_indicator(recent["close"], window=ema_window)

    last = recent.iloc[-1]
    prev_close = recent["close"].iloc[-2]

    drop = (prev_close - last["close"]) / prev_close if prev_close else 0.0
    vol_ok = (
        pd.notna(vol_ma.iloc[-1])
        and vol_ma.iloc[-1] > 0
        and last["volume"] >= vol_ma.iloc[-1] * vol_mult
    )
    ema_ok = pd.isna(ema.iloc[-1]) or last["close"] < ema.iloc[-1]

    if drop >= drop_pct and vol_ok and ema_ok:
        score = min(drop / drop_pct, 1.0)
        if atr_norm:
            score = normalize_score_by_volatility(df, score)
        score = max(0.0, min(float(score), 1.0))
        return score, "long"

    return 0.0, "none"


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
