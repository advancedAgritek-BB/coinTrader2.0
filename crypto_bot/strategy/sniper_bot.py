from typing import Dict, Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.volatility_filter import calc_atr
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

DEFAULT_PAIRS = ["BTC/USD", "ETH/USD"]
ALLOWED_PAIRS = load_liquid_pairs() or DEFAULT_PAIRS

logger = setup_logger(__name__, LOG_DIR / "bot.log")

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("sniper_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Dict[str, float | int | str]] = None,
    *,
    breakout_pct: float = 0.01,
    volume_multiple: float = 1.2,
    max_history: int = 30,
    initial_window: int = 3,
    min_volume: float = 100.0,
    direction: str = "auto",
    high_freq: bool = False,
    atr_window: int = 14,
    volume_window: int = 5,
    price_fallback: bool = True,
    fallback_atr_mult: float = 1.5,
    fallback_volume_mult: float = 1.2,
) -> Tuple[float, str, float, bool]:
    """Detect pumps for newly listed tokens using early price and volume action.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data ordered oldest -> newest.
    config : dict, optional
        Configuration values overriding the keyword defaults.
    breakout_pct : float, optional
        Minimum percent change from the first close considered a breakout.
    volume_multiple : float, optional
        Minimum multiple of the average volume of the first ``initial_window``
        candles considered abnormal.
    max_history : int, optional
        Maximum history length still considered a new listing.
    initial_window : int, optional
        Number of early candles used to compute baseline volume.
    min_volume : float, optional
        Minimum trade volume for the latest candle to consider a signal.
    direction : {"auto", "long", "short"}, optional
        Force a trade direction or infer automatically.
    high_freq : bool, optional
        When ``True`` the function expects 1m candles and shortens
        ``max_history`` and ``initial_window`` so signals can trigger
        right after a listing.
    atr_window : int, optional
        Window length used to compute ATR for event detection.
    volume_window : int, optional
        Window length used to compute average volume for event detection.
    price_fallback : bool, optional
        Enable ATR based fallback when breakout conditions fail.
        Defaults to ``True``.
    fallback_atr_mult : float, optional
        Required candle body multiple of ATR for the fallback.
        Defaults to ``1.5``.
    fallback_volume_mult : float, optional
        Required volume multiple for the fallback.
        Defaults to ``1.2``.

    Returns
    -------
    Tuple[float, str, float, bool]
        Score between 0 and 1, trade direction, ATR value and event flag.
    """
    symbol = config.get("symbol", "unknown") if config else "unknown"
    if (
        symbol
        and symbol != "unknown"
        and ALLOWED_PAIRS
        and symbol not in ALLOWED_PAIRS
    ):
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none", 0.0, False

    if config:
        breakout_pct = config.get("breakout_pct", breakout_pct)
        volume_multiple = config.get("volume_multiple", volume_multiple)
        max_history = config.get("max_history", max_history)
        initial_window = config.get("initial_window", initial_window)
        min_volume = config.get("min_volume", min_volume)
        direction = config.get("direction", direction)
        atr_window = int(config.get("atr_window", atr_window))
        volume_window = int(config.get("volume_window", volume_window))
        price_fallback = config.get("price_fallback", price_fallback)
        fallback_atr_mult = config.get("fallback_atr_mult", fallback_atr_mult)
        fallback_volume_mult = config.get("fallback_volume_mult", fallback_volume_mult)

    if high_freq:
        max_history = min(max_history, 20)
        initial_window = max(1, initial_window // 2)

    if len(df) < initial_window:
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none", 0.0, False

    first_close = df["close"].iloc[0]
    if first_close == 0:
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none", 0.0, False

    price_change = df["close"].iloc[-1] / first_close - 1
    if direction == "auto" and price_change < 0:
        atr = calc_atr(df)
        score = 1.0
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        logger.info("Signal for %s: %s, %s", symbol, score, "short")
        return score, "short", atr, False

    base_volume = df["volume"].iloc[:initial_window].mean()
    vol_ratio = df["volume"].iloc[-1] / base_volume if base_volume > 0 else 0

    atr_window = min(atr_window, len(df))
    atr_series = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window
    )
    atr = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

    if len(df) > volume_window:
        prev_vol = df["volume"].iloc[-(volume_window + 1):-1]
    else:
        prev_vol = df["volume"].iloc[:-1]
    avg_vol = prev_vol.mean() if not prev_vol.empty else 0.0
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    event = False
    if atr > 0 and avg_vol > 0:
        if body >= 2 * atr and df["volume"].iloc[-1] >= 2 * avg_vol:
            event = True

    if df["volume"].iloc[-1] < min_volume:
        logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none", atr, event

    if (
        len(df) <= max_history
        and abs(price_change) >= breakout_pct
        and vol_ratio >= volume_multiple
    ):
        price_score = min(abs(price_change) / breakout_pct, 1.0)
        vol_score = min(vol_ratio / volume_multiple, 1.0)
        score = (price_score + vol_score) / 2
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        if direction not in {"auto", "long", "short"}:
            direction = "auto"
        trade_direction = direction
        if direction == "auto":
            trade_direction = "short" if price_change < 0 else "long"
        logger.info("Signal for %s: %s, %s", symbol, score, trade_direction)
        return score, trade_direction, atr, event

    trade_direction = direction
    score = 0.0

    if price_fallback:
        atr_series = calc_atr(df)
        if len(atr_series) == 0 or atr_series.iloc[-1] <= 0:
            logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
            return 0.0, "none", 0.0, event
        atr = float(atr_series.iloc[-1])
        body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
        avg_vol = df["volume"].iloc[:-1].mean()
        if (
            body > atr * fallback_atr_mult
            and avg_vol > 0
            and df["volume"].iloc[-1] > avg_vol * fallback_volume_mult
        ):
            score = 1.0
            if MODEL is not None:
                try:  # pragma: no cover - best effort
                    ml_score = MODEL.predict(df)
                    score = (score + ml_score) / 2
                except Exception:
                    pass
            if config is None or config.get("atr_normalization", True):
                score = normalize_score_by_volatility(df, score)
            if direction not in {"auto", "long", "short"}:
                direction = "auto"
            trade_direction = direction
            if direction == "auto":
                trade_direction = (
                    "short"
                    if df["close"].iloc[-1] < df["open"].iloc[-1]
                    else "long"
                )
            logger.info("Signal for %s: %s, %s", symbol, score, trade_direction)
            return score, trade_direction, atr, event

    logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
    return 0.0, "none", atr, event


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
