from typing import Tuple

import pandas as pd
import ta
from ta.trend import ADXIndicator

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils import stats
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.cooldown_manager import cooldown, in_cooldown
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = setup_logger(__name__, LOG_DIR / "bot.log")

try:  # Optional LightGBM integration
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("dip_hunter")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    higher_df: pd.DataFrame | None = None,
    config: dict | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Tuple[float, str]:
    """Detect deep dips for mean reversion long entries."""
    symbol = config.get("symbol", "") if config else ""
    params = config.get("dip_hunter", {}) if config else {}
    cooldown_enabled = bool(params.get("cooldown_enabled", False))
    strategy = "dip_hunter"

    if cooldown_enabled and symbol and in_cooldown(symbol, strategy):
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "cooldown")
        return 0.0, "none"

    if len(df) < 50:
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none"

    rsi_window = int(params.get("rsi_window", 14))
    rsi_oversold = float(params.get("rsi_oversold", 30.0))
    dip_pct = float(params.get("dip_pct", 0.03))
    dip_bars = int(params.get("dip_bars", 3))
    vol_window = int(params.get("vol_window", 20))
    vol_mult = float(params.get("vol_mult", 1.5))
    adx_window = int(params.get("adx_window", 14))
    adx_threshold = float(params.get("adx_threshold", 25.0))
    bb_window = int(params.get("bb_window", 20))
    ema_trend = int(params.get("ema_trend", 200))
    ml_weight = float(params.get("ml_weight", 0.5))
    atr_normalization = bool(params.get("atr_normalization", True))

    lookback = max(rsi_window, vol_window, adx_window, bb_window, dip_bars)
    recent = df.iloc[-(lookback + 1) :]

    rsi = ta.momentum.rsi(recent["close"], window=rsi_window)
    adx = ADXIndicator(recent["high"], recent["low"], recent["close"], window=adx_window).adx()
    bb = ta.volatility.BollingerBands(recent["close"], window=bb_window)
    bb_pct = bb.bollinger_pband()
    vol_ma = recent["volume"].rolling(vol_window).mean()

    rsi = cache_series("rsi_dip", df, rsi, lookback)
    adx = cache_series("adx_dip", df, adx, lookback)
    bb_pct = cache_series("bb_pct_dip", df, bb_pct, lookback)
    vol_ma = cache_series("vol_ma_dip", df, vol_ma, lookback)

    recent = recent.copy()
    recent["rsi"] = rsi
    recent["adx"] = adx
    recent["bb_pct"] = bb_pct
    recent["vol_ma"] = vol_ma

    latest = recent.iloc[-1]

    if len(recent) < dip_bars + 1:
        return 0.0, "none"
    recent_returns = recent["close"].pct_change().iloc[-dip_bars:]
    dip_size = recent_returns.sum()
    is_dip = dip_size <= -dip_pct

    with cooldown(symbol, strategy) as cd:
        if cooldown_enabled and symbol and not cd.allowed:
            logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
            return 0.0, "none"

        oversold = latest["rsi"] < rsi_oversold and latest["bb_pct"] < 0
        vol_spike = (
            latest["volume"] > latest["vol_ma"] * vol_mult if latest["vol_ma"] > 0 else False
        )

        range_bound = latest["adx"] < adx_threshold
        if higher_df is not None and not higher_df.empty:
            h_lookback = max(ema_trend, 1)
            h_recent = higher_df.iloc[-(h_lookback + 1) :]
            ema_h = ta.trend.ema_indicator(h_recent["close"], window=ema_trend)
            ema_h = cache_series("ema_trend_h", higher_df, ema_h, h_lookback)
            in_trend = higher_df["close"].iloc[-1] > ema_h.iloc[-1]
        else:
            # Default to a neutral stance when higher timeframe data is unavailable
            in_trend = True

        favorable_regime = range_bound or in_trend

        if is_dip and oversold and vol_spike and favorable_regime:
            dip_score = min(abs(dip_size) / dip_pct, 1.0)
            oversold_score = min((rsi_oversold - latest["rsi"]) / rsi_oversold, 1.0)
            vol_z = stats.zscore(recent["volume"], vol_window).iloc[-1]
            vol_score = min(max(vol_z / 2, 0), 1.0)
            score = dip_score * 0.4 + oversold_score * 0.3 + vol_score * 0.3

            if MODEL:
                try:  # pragma: no cover - best effort
                    ml_score = MODEL.predict(df)
                    score = score * (1 - ml_weight) + ml_score * ml_weight
                except Exception:
                    pass

            if atr_normalization:
                score = normalize_score_by_volatility(df, score)

            score = max(0.0, min(score, 1.0))
            logger.info("Signal for %s: %s, %s", symbol, score, "long")
            if cooldown_enabled and symbol:
                cd.mark()
            return score, "long"

    return 0.0, "none"


class regime_filter:
    """Match mean-reverting regime for Dip Hunter."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
