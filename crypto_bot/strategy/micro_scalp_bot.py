from typing import Optional, Tuple

import asyncio
import logging
import pandas as pd
import ta
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
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

if ML_AVAILABLE:
    MODEL = load_model("micro_scalp_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def _wick_ratios(row: pd.Series) -> Tuple[float, float]:
    """Return lower and upper wick ratios for a candle.

    The ratios are calculated relative to the total candle range.
    """

    high = row["high"]
    low = row["low"]
    open_ = row["open"]
    close = row["close"]

    rng = high - low
    if rng <= 0:
        return 0.0, 0.0

    body_low = min(open_, close)
    body_high = max(open_, close)

    lower = body_low - low
    upper = high - body_high

    return lower / rng, upper / rng


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    higher_df: pd.DataFrame | None = None,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[dict] = None,
    tick_data: pd.DataFrame | None = None,
    book: Optional[dict] = None,
    ticks: Optional[pd.DataFrame] = None,
) -> Tuple[float, str]:
    """Return short-term signal using EMA crossover on 1m data.

    Parameters
    ----------
    df : pd.DataFrame
        Minute level OHLCV data.
    config : dict, optional
        Optional configuration overriding defaults located under
        ``micro_scalp_bot`` in ``config.yaml``.
        Supports ``wick_pct`` for a symmetric wick requirement or
        ``lower_wick_pct``/``upper_wick_pct`` for individual thresholds.
    higher_df : pd.DataFrame, optional
        Higher timeframe OHLCV data used to confirm the trend. When provided
        the function only returns a signal if ``trend_fast`` is above
        ``trend_slow`` for longs (and vice versa for shorts).
    book : dict, optional
        Order book data with ``bids`` and ``asks`` arrays.
    ticks : pd.DataFrame, optional
        Optional tick-level data with ``price`` and optional ``volume`` columns.
    mempool_monitor : SolanaMempoolMonitor, optional
        Instance used to monitor the Solana priority fee.
    mempool_cfg : dict, optional
        Configuration controlling the mempool fee check.
    
    Additional config options
    -------------------------
    ``imbalance_ratio``
        Order book imbalance threshold used to filter or penalize signals.
    ``imbalance_penalty``
        Factor applied to the score when imbalance is against the trade.
    ``imbalance_filter``
        If ``false``, skip the order book imbalance check.
    ``trend_filter``
        When ``false`` the higher timeframe EMA confirmation is ignored.
    """
    if tick_data is not None and not tick_data.empty:
        df = pd.concat(
            [df.reset_index(drop=True), tick_data.reset_index(drop=True)],
            ignore_index=True,
        )

    if df.empty:
        return 0.0, "none"

    if ticks is not None and not ticks.empty:
        price_col = "price" if "price" in ticks.columns else "close"
        vol = ticks["volume"] if "volume" in ticks.columns else 0
        tick_df = pd.DataFrame(
            {
                "open": ticks[price_col],
                "high": ticks[price_col],
                "low": ticks[price_col],
                "close": ticks[price_col],
                "volume": vol,
            }
        )
        df = pd.concat(
            [df.reset_index(drop=True), tick_df.reset_index(drop=True)],
            ignore_index=True,
        )

    params = config.get("micro_scalp_bot", {}) if config else {}

    cfg = mempool_cfg or {}
    if mempool_monitor and cfg.get("enabled"):
        threshold = cfg.get("suspicious_fee_threshold", 0.0)
        try:
            suspicious = asyncio.run(mempool_monitor.is_suspicious(threshold))
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                suspicious = loop.run_until_complete(
                    mempool_monitor.is_suspicious(threshold)
                )
            except Exception:
                suspicious = False
        except Exception:
            suspicious = False
        if suspicious:
            return 0.0, "none"
    fast_window = int(params.get("ema_fast", 2))
    slow_window = int(params.get("ema_slow", 5))
    vol_window = int(params.get("volume_window", 20))
    min_vol_z = float(params.get("min_vol_z", -0.5))
    atr_period = int(params.get("atr_period", 14))
    min_atr_pct = float(params.get("min_atr_pct", 0.001))
    min_momentum_pct = float(params.get("min_momentum_pct", 0))
    wick_pct = float(params.get("wick_pct", 0))
    lower_wick_pct = float(params.get("lower_wick_pct", wick_pct))
    upper_wick_pct = float(params.get("upper_wick_pct", wick_pct))
    confirm_bars = int(params.get("confirm_bars", 0))
    fresh_cross_only = bool(params.get("fresh_cross_only", False))
    imbalance_ratio = float(params.get("imbalance_ratio", 0))
    imbalance_penalty = float(params.get("imbalance_penalty", 0.2))
    imbalance_filter = bool(params.get("imbalance_filter", True))
    trend_fast = int(params.get("trend_fast", 0))
    trend_slow = int(params.get("trend_slow", 0))
    _ = params.get("trend_timeframe")
    trend_filter = bool(params.get("trend_filter", True))

    if len(df) < slow_window:
        return 0.0, "none"

    lookback = max(slow_window, vol_window, atr_period)

    ema_fast_full = ta.trend.ema_indicator(df["close"], window=fast_window)
    ema_slow_full = ta.trend.ema_indicator(df["close"], window=slow_window)
    ema_fast = cache_series("ema_fast", df, ema_fast_full, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow_full, lookback)

    atr_window = min(atr_period, len(df))
    atr_series_full = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window
    )
    atr_series = cache_series(f"atr_{atr_period}", df, atr_series_full, lookback)

    recent = df.iloc[-(lookback + 1) :].copy()
    recent["ema_fast"] = ema_fast
    recent["ema_slow"] = ema_slow
    recent["atr"] = atr_series
    recent["momentum"] = recent["ema_fast"] - recent["ema_slow"]
    df = recent

    latest = df.iloc[-1]
    lower_wick_ratio, upper_wick_ratio = _wick_ratios(latest)
    if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
        return 0.0, "none"

    if min_atr_pct and "atr" in df.columns and latest["close"] > 0:
        if pd.isna(latest["atr"]) or latest["atr"] / latest["close"] < min_atr_pct:
            return 0.0, "none"

    if min_vol_z and "volume" in df.columns:
        vol_ma_series = df["volume"].rolling(vol_window).mean()
        vol_std_series = df["volume"].rolling(vol_window).std()
        vol_ma_series = cache_series("volume_ma", df, vol_ma_series, lookback)
        vol_std_series = cache_series("volume_std", df, vol_std_series, lookback)
        vol_mean = vol_ma_series.iloc[-1]
        vol_std = vol_std_series.iloc[-1]
        vol_z = (latest["volume"] - vol_mean) / vol_std if vol_std > 0 else float("-inf")
        if pd.isna(vol_z) or vol_z < min_vol_z:
            return 0.0, "none"

    trend_fast_val = None
    trend_slow_val = None
    if higher_df is not None and trend_fast and trend_slow:
        trend_lookback = max(trend_fast, trend_slow)
        h_recent = higher_df.iloc[-(trend_lookback + 1) :]
        t_fast = ta.trend.ema_indicator(h_recent["close"], window=trend_fast)
        t_slow = ta.trend.ema_indicator(h_recent["close"], window=trend_slow)
        t_fast = cache_series("trend_fast", higher_df, t_fast, trend_lookback)
        t_slow = cache_series("trend_slow", higher_df, t_slow, trend_lookback)
        trend_fast_val = t_fast.iloc[-1]
        trend_slow_val = t_slow.iloc[-1]
        if pd.isna(trend_fast_val) or pd.isna(trend_slow_val):
            return 0.0, "none"

    momentum = df["momentum"].iloc[-1]
    if momentum == 0:
        return 0.0, "none"

    if min_momentum_pct and abs(momentum) / latest["close"] < min_momentum_pct:
        return 0.0, "none"

    if confirm_bars > 0:
        if len(df) < confirm_bars:
            return 0.0, "none"
        signs = (df["momentum"].iloc[-confirm_bars:].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))
        if signs.abs().min() == 0 or not (signs == signs.iloc[-1]).all():
            return 0.0, "none"
        if fresh_cross_only and len(df) >= confirm_bars + 1:
            prev_sign = 1 if df["momentum"].iloc[-confirm_bars - 1] > 0 else -1 if df["momentum"].iloc[-confirm_bars - 1] < 0 else 0
            if prev_sign == signs.iloc[-1]:
                return 0.0, "none"
        elif fresh_cross_only and len(df) < confirm_bars + 1:
            return 0.0, "none"

    score = min(abs(momentum) / latest["close"], 1.0)
    if score > 0 and MODEL is not None:
        try:  # pragma: no cover - best effort
            ml_score = MODEL.predict(df)
            score = (score + ml_score) / 2
        except Exception:
            pass
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if momentum > 0 else "short"

    if direction == "long" and lower_wick_ratio < lower_wick_pct:
        return 0.0, "none"
    if direction == "short" and upper_wick_ratio < upper_wick_pct:
        return 0.0, "none"

    book_data = book or params.get("order_book")
    if (
        isinstance(book_data, dict)
        and book_data.get("bids")
        and book_data.get("asks")
    ):
        bids_list = book_data["bids"]
        asks_list = book_data["asks"]
        best_bid = bids_list[0][0]
        best_ask = asks_list[0][0]
        mid_price = (best_bid + best_ask) / 2
        if mid_price > 0 and (best_ask - best_bid) / mid_price > 0.003:
            return 0.0, "none"

    if (
        imbalance_filter
        and imbalance_ratio
        and isinstance(book_data, dict)
        and book_data.get("bids")
        and book_data.get("asks")
    ):
        bids = sum(v for _, v in book_data["bids"])
        asks = sum(v for _, v in book_data["asks"])
        if bids and asks:
            imbalance = bids / asks
            if direction == "long" and imbalance < imbalance_ratio:
                if imbalance_penalty > 0:
                    score *= imbalance_penalty
                else:
                    return 0.0, "none"
            if direction == "short" and imbalance > imbalance_ratio:
                if imbalance_penalty > 0:
                    score *= imbalance_penalty
                else:
                    return 0.0, "none"

    if trend_filter and trend_fast_val is not None and trend_slow_val is not None:
        if direction == "long" and trend_fast_val <= trend_slow_val:
            return 0.0, "none"
        if direction == "short" and trend_fast_val >= trend_slow_val:
            return 0.0, "none"

    if mempool_monitor is not None:
        try:
            fee = asyncio.run(mempool_monitor.fetch_priority_fee())
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                fee = loop.run_until_complete(mempool_monitor.fetch_priority_fee())
            except Exception:
                fee = None
        except Exception:
            fee = None
        if fee is not None and fee < 5:
            score *= 1.2

    return score, direction


class regime_filter:
    """Match scalp regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "scalp"


class Strategy:
    """Strategy wrapper so :func:`load_strategies` can auto-register it."""

    def __init__(self) -> None:
        self.name = "micro_scalp_bot"
        self.generate_signal = generate_signal
        self.regime_filter = regime_filter


__all__ = ["generate_signal", "regime_filter", "Strategy"]
