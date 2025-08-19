"""Statistical arbitrage pair trading strategy."""

from typing import Optional, Tuple

import logging
import pandas as pd

from crypto_bot.utils import stats
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

NAME = "stat_arb_bot"
if ML_AVAILABLE:
    MODEL = load_model("stat_arb_bot")
else:  # pragma: no cover - fallback
    MODEL = None

_ZSCORE_THRESHOLD_DEFAULT = 2.0
_LOOKBACK_DEFAULT = 20
_CORRELATION_THRESHOLD = 0.8


def generate_signal(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: Optional[dict] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Return (score, direction) based on the price spread z-score."""
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return 0.0, "none"

    threshold = float(config.get("zscore_threshold", _ZSCORE_THRESHOLD_DEFAULT)) if config else _ZSCORE_THRESHOLD_DEFAULT
    lookback = int(config.get("lookback", _LOOKBACK_DEFAULT)) if config else _LOOKBACK_DEFAULT

    if len(df_a) < lookback or len(df_b) < lookback:
        return 0.0, "none"

    corr = df_a["close"].corr(df_b["close"])
    if pd.isna(corr) or corr < _CORRELATION_THRESHOLD:
        return 0.0, "none"

    spread = df_a["close"] - df_b["close"]
    z = stats.zscore(spread, lookback)
    z = cache_series("stat_arb_z", df_a, z, lookback)
    if z.empty:
        return 0.0, "none"

    z_last = z.iloc[-1]
    if abs(z_last) <= threshold:
        return 0.0, "none"

    direction = "long" if z_last < 0 else "short"
    score = float(abs(z_last))

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df_a)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df_a, score)

    return score, direction


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
