from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_bot.utils import volatility
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once
NAME = "sniper_bot"
DEFAULT_PAIRS = ["BTC/USD", "ETH/USD"]
ALLOWED_PAIRS = load_liquid_pairs() or DEFAULT_PAIRS

logger = setup_logger(__name__, LOG_DIR / "bot.log")
# Shared logger for symbol scoring
score_logger = setup_logger(
    "symbol_filter", LOG_DIR / "symbol_filter.log", to_console=False
)

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
    symbol: str | None = None,
    timeframe: str | None = None,
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
    **kwargs,
) -> Tuple[float, str, float | dict, bool]:
    """Detect pumps for newly listed tokens using early price and volume action.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data ordered oldest -> newest.
    config : dict, optional
        Configuration values overriding the keyword defaults.
    symbol : str, optional
        Asset symbol for the provided data.
    timeframe : str, optional
        Candle timeframe for ``df``.
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
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    breakout_pct = kwargs.get("breakout_pct", 0.01)
    volume_multiple = kwargs.get("volume_multiple", 1.2)
    max_history = kwargs.get("max_history", 30)
    initial_window = kwargs.get("initial_window", 3)
    min_volume = kwargs.get("min_volume", 100.0)
    direction = kwargs.get("direction", "auto")
    high_freq = kwargs.get("high_freq", False)
    atr_window = kwargs.get("atr_window", 14)
    volume_window = kwargs.get("volume_window", 5)
    price_fallback = kwargs.get("price_fallback", True)
    fallback_atr_mult = kwargs.get("fallback_atr_mult", 1.5)
    fallback_volume_mult = kwargs.get("fallback_volume_mult", 1.2)

    symbol = symbol or (config.get("symbol", "unknown") if config else "unknown")
    timeframe = timeframe or (config.get("timeframe") if config else None)
    if (
        symbol
        and symbol != "unknown"
        and ALLOWED_PAIRS
        and symbol not in ALLOWED_PAIRS
    ):
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
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
        msg = "Signal for %s:%s -> %.3f, %s"
        score_logger.info(msg, symbol or "unknown", timeframe or "N/A", 0.0, "none")
        logger.info(msg, symbol or "unknown", timeframe or "N/A", 0.0, "none")
        return 0.0, "none", 0.0, False

    first_close = df["close"].iloc[0]
    if first_close == 0:
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
        return 0.0, "none", 0.0, False

    price_change = df["close"].iloc[-1] / first_close - 1
    if direction == "auto" and price_change < 0:
        atr_window = min(atr_window, len(df))
        atr_series = volatility.calc_atr(df, period=atr_window)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
        if not np.isfinite(atr) or atr <= 0.0:
            score_logger.info(
                "Signal for %s:%s -> %.3f, %s",
                symbol or "unknown",
                timeframe or "N/A",
                0.0,
                "none",
            )
            return 0.0, "none", {"reason": "bad_atr"}, False
        score = 1.0
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = volatility.normalize_score_by_volatility(df, score)
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            score,
            "short",
        )
        return score, "short", float(atr), False

    base_volume = df["volume"].iloc[:initial_window].mean()
    vol_ratio = df["volume"].iloc[-1] / base_volume if base_volume > 0 else 0

    atr_window = min(atr_window, len(df))
    atr_series = volatility.calc_atr(df, period=atr_window)
    atr_last = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
    atr = atr_last
    event = False
    if not np.isfinite(atr_last) or atr_last <= 0.0:
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
        return 0.0, "none", {"reason": "bad_atr"}, event

    if len(df) > volume_window:
        prev_vol = df["volume"].iloc[-(volume_window + 1):-1]
    else:
        prev_vol = df["volume"].iloc[:-1]
    avg_vol = prev_vol.mean() if not prev_vol.empty else 0.0
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    if avg_vol > 0:
        atr_val = atr_last
        if atr_val > 0 and body >= 2 * atr_val and df["volume"].iloc[-1] >= 2 * avg_vol:
            event = True

    if df["volume"].iloc[-1] < min_volume:
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
        return 0.0, "none", float(atr_last), event

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
            score = volatility.normalize_score_by_volatility(df, score)
        if direction not in {"auto", "long", "short"}:
            direction = "auto"
        trade_direction = direction
        if direction == "auto":
            trade_direction = "short" if price_change < 0 else "long"
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            score,
            trade_direction,
        )
        return score, trade_direction, float(atr), event

    trade_direction = direction
    score = 0.0

    if price_fallback:
        atr_series = volatility.calc_atr(df, period=atr_window)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
        if not np.isfinite(atr) or atr <= 0.0:
            score_logger.info(
                "Signal for %s:%s -> %.3f, %s",
                symbol or "unknown",
                timeframe or "N/A",
                0.0,
                "none",
            )
            return 0.0, "none", {"reason": "bad_atr"}, event
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
                score = volatility.normalize_score_by_volatility(df, score)
            if direction not in {"auto", "long", "short"}:
                direction = "auto"
            trade_direction = direction
            if direction == "auto":
                trade_direction = (
                    "short"
                    if df["close"].iloc[-1] < df["open"].iloc[-1]
                    else "long"
                )
            score_logger.info(
                "Signal for %s:%s -> %.3f, %s",
                symbol or "unknown",
                timeframe or "N/A",
                score,
                trade_direction,
            )
            return score, trade_direction, atr, event

    score_logger.info(
        "Signal for %s:%s -> %.3f, %s",
        symbol or "unknown",
        timeframe or "N/A",
        0.0,
        "none",
    )
    return 0.0, "none", float(atr) if atr is not None else 0.0, event


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
