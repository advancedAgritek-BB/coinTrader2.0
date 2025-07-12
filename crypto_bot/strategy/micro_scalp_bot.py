from typing import Optional, Tuple

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


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
    book: Optional[dict] = None,
    ticks: Optional[pd.DataFrame] = None,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return short-term signal using EMA crossover on 1m data.

    Parameters
    ----------
    df : pd.DataFrame
        Minute level OHLCV data.
    config : dict, optional
        Optional configuration overriding defaults located under
        ``micro_scalp`` in ``config.yaml``.
        Supports ``wick_pct`` for a symmetric wick requirement or
        ``lower_wick_pct``/``upper_wick_pct`` for individual thresholds.
    higher_df : pd.DataFrame, optional
        Higher timeframe OHLCV data used to confirm the trend. When provided
        the function only returns a signal if ``trend_fast`` is above
        ``trend_slow`` for longs (and vice versa for shorts).
    ticks
        Optional tick-level data with ``price`` and optional ``volume`` columns.
    mempool_monitor
        Optional :class:`~crypto_bot.execution.solana_mempool.SolanaMempoolMonitor`
        instance to check the current priority fee.
    mempool_cfg
        Configuration dictionary with ``enabled`` and ``suspicious_fee_threshold``
        keys controlling the fee check.
    """
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
        df = pd.concat([df, tick_df], ignore_index=True)

    params = config.get("micro_scalp", {}) if config else {}

    cfg = mempool_cfg or {}
    if mempool_monitor and cfg.get("enabled"):
        threshold = cfg.get("suspicious_fee_threshold", 0.0)
        if mempool_monitor.is_suspicious(threshold):
            return 0.0, "none"
    fast_window = int(params.get("ema_fast", 3))
    slow_window = int(params.get("ema_slow", 8))
    vol_window = int(params.get("volume_window", 20))
    min_vol_z = float(params.get("min_vol_z", 0))
    atr_period = int(params.get("atr_period", 14))
    min_atr_pct = float(params.get("min_atr_pct", 0))
    min_momentum_pct = float(params.get("min_momentum_pct", 0))
    wick_pct = float(params.get("wick_pct", 0))
    lower_wick_pct = float(params.get("lower_wick_pct", wick_pct))
    upper_wick_pct = float(params.get("upper_wick_pct", wick_pct))
    confirm_bars = int(params.get("confirm_bars", 1))
    fresh_cross_only = bool(params.get("fresh_cross_only", True))
    imbalance_ratio = float(params.get("imbalance_ratio", 0))
    imbalance_penalty = float(params.get("imbalance_penalty", 0))
    trend_fast = int(params.get("trend_fast", 0))
    trend_slow = int(params.get("trend_slow", 0))
    _ = params.get("trend_timeframe")

    if len(df) < slow_window:
        return 0.0, "none"

    lookback = max(slow_window, vol_window, atr_period)
    recent = df.iloc[-(lookback + 1) :]

    ema_fast = ta.trend.ema_indicator(recent["close"], window=fast_window)
    ema_slow = ta.trend.ema_indicator(recent["close"], window=slow_window)

    ema_fast = cache_series("ema_fast", df, ema_fast, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow, lookback)
    atr_window = min(atr_period, len(recent))
    atr_series = ta.volatility.average_true_range(
        recent["high"], recent["low"], recent["close"], window=atr_window
    )
    atr_series = cache_series(f"atr_{atr_period}", df, atr_series, lookback)

    df = recent.copy()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow
    df["atr"] = atr_series
    df["momentum"] = df["ema_fast"] - df["ema_slow"]

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
        vol_z = (latest["volume"] - vol_mean) / vol_std if vol_std > 0 else 0.0
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

    if min_momentum_pct:
        momentum_pct = abs(momentum) / latest["close"]
        momentum_change = abs(df["momentum"].pct_change().iloc[-1])
        if momentum_pct < min_momentum_pct or momentum_change < min_momentum_pct:
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
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if momentum > 0 else "short"

    if direction == "long" and lower_wick_ratio < lower_wick_pct:
        return 0.0, "none"
    if direction == "short" and upper_wick_ratio < upper_wick_pct:
        return 0.0, "none"

    book_data = book or params.get("order_book")
    if imbalance_ratio and isinstance(book_data, dict):
        bids = sum(v for _, v in book_data.get("bids", []))
        asks = sum(v for _, v in book_data.get("asks", []))
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

    if trend_fast_val is not None and trend_slow_val is not None:
        if direction == "long" and trend_fast_val <= trend_slow_val:
            return 0.0, "none"
        if direction == "short" and trend_fast_val >= trend_slow_val:
            return 0.0, "none"

    return score, direction
