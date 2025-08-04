from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("momentum_bot")
except Exception:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_mult = float(params.get("volume_mult", 1.5))

    lookback = max(window, vol_window)
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]

    dc_high = recent["high"].rolling(window).max().shift(1)
    dc_low = recent["low"].rolling(window).min().shift(1)
    vol_ma = recent["volume"].rolling(vol_window).mean()

    dc_high = cache_series("momentum_dc_high", df, dc_high, lookback)
    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma

    close = recent["close"].iloc[-1]
    volume = recent["volume"].iloc[-1]

    long_cond = close > dc_high.iloc[-1]
    short_cond = close < dc_low.iloc[-1]
    vol_ok = vol_ma.iloc[-1] > 0 and volume > vol_ma.iloc[-1] * vol_mult

    score = 0.0
    direction = "none"
    if long_cond and vol_ok:
        score = 1.0
        direction = "long"
    elif short_cond and vol_ok:
        score = 1.0
        direction = "short"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match trending and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}
