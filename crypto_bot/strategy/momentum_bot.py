from typing import Optional, Tuple

import logging
import pandas as pd
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False

MODEL: Optional[object]
if ML_AVAILABLE:
    MODEL = load_model("momentum_bot")
else:  # pragma: no cover - fallback
    MODEL = None
    warn_ml_unavailable_once()


def generate_signal(
    df: pd.DataFrame,
    config: dict | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_mult = float(params.get("volume_mult", 1.5))

    if len(df) < vol_window:
        return 0.0, "none"

    lookback = min(len(df), max(window, vol_window))
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

    latest = recent.iloc[-1]

    long_cond = latest["close"] > dc_high.iloc[-1]
    short_cond = latest["close"] < dc_low.iloc[-1]
    vol_ok = (
        pd.notna(latest["vol_ma"])
        and latest["vol_ma"] > 0
        and latest["volume"] > latest["vol_ma"] * vol_mult
    )

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
