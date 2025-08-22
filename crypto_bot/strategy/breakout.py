from __future__ import annotations

from typing import Optional, Tuple

import logging
import pandas as pd
import ta
from crypto_bot.utils.config_helpers import short_selling_enabled

from crypto_bot.utils import stats
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.market_loader import get_progress_eta

NAME = "breakout"
logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str, float]:
    """Selective breakout strategy with compression and volume filters.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV data on a 5m timeframe.
    config : dict, optional
        Configuration containing a ``breakout`` section with parameters.
    symbol : str, optional
        Asset symbol for the data. Unused but accepted for compatibility.
    timeframe : str, optional
        Candle timeframe for ``df``. Unused but accepted for compatibility.
    **kwargs : dict
        Additional keyword arguments for forward compatibility.

    Returns
    -------
    Tuple[float, str, float]
        ``(score, direction, atr)`` where ``score`` is 1 on signal, 0 otherwise.
    """
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    cfg_all = config or {}
    cfg = cfg_all.get("breakout", cfg_all)
    donchian_len = int(cfg.get("donchian_len", 20))
    keltner_len = int(cfg.get("keltner_len", 20))
    keltner_mult = float(cfg.get("keltner_mult", 1.5))
    bbw_lookback = int(cfg.get("bbw_lookback", 100))
    bbw_pct_max = float(cfg.get("bbw_pct_max", 20.0))
    vol_window = int(cfg.get("vol_window", 20))
    volume_z_min = float(cfg.get("volume_z_min", 1.0))
    atr_len = int(cfg.get("atr_len", keltner_len))
    max_spread_bp = float(cfg.get("max_spread_bp", 5.0))
    shorting = short_selling_enabled(cfg_all)
    score_threshold = float(cfg.get("score_threshold", 0.0))
    lookback = max(donchian_len, keltner_len, bbw_lookback, vol_window, atr_len)
    progress_logging = bool(cfg_all.get("ohlcv", {}).get("progress_logging", False))
    required = lookback + 1

    if df is None or df.empty:
        msg = "signal=breakout side=none reason=insufficient_data vol_z=nan bbw_pct=nan"
        if progress_logging and symbol and timeframe:
            pct, eta = get_progress_eta(symbol, timeframe, required)
            msg += f" progress={pct:.1f}% eta={eta:.1f}s"
        logger.info(msg)
        return 0.0, "none", 0.0

    if len(df) < required:
        msg = "signal=breakout side=none reason=insufficient_data vol_z=nan bbw_pct=nan"
        if progress_logging and symbol and timeframe:
            pct, eta = get_progress_eta(symbol, timeframe, required)
            msg += f" progress={pct:.1f}% eta={eta:.1f}s"
        logger.info(msg)
        return 0.0, "none", 0.0

    recent = df.iloc[-(lookback + 1) :].copy()
    close = recent["close"]
    high = recent["high"]
    low = recent["low"]
    volume = recent["volume"]
    spread_bp = float(recent.get("spread_bp", pd.Series(0, index=recent.index)).iloc[-1])

    dc_high = high.rolling(donchian_len).max().shift(1)
    dc_low = low.rolling(donchian_len).min().shift(1)

    ema = close.ewm(span=keltner_len, adjust=False).mean()
    atr_keltner = ta.volatility.average_true_range(high, low, close, window=keltner_len)
    kc_upper = ema + atr_keltner * keltner_mult
    kc_lower = ema - atr_keltner * keltner_mult

    bb = ta.volatility.BollingerBands(close, window=keltner_len)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    bb_width = cache_series("breakout_bb_width", df, bb_width, lookback)
    bbw_series = bb_width.iloc[-bbw_lookback:]
    bbw_pct = bbw_series.rank(pct=True).iloc[-1] * 100 if not bbw_series.empty else float("nan")

    vol_z_series = stats.zscore(volume, vol_window)
    vol_z = vol_z_series.iloc[-1] if not vol_z_series.empty else float("nan")

    atr = ta.volatility.average_true_range(high, low, close, window=atr_len)
    atr_latest = float(atr.iloc[-1]) if not atr.empty else 0.0

    filters_ok = (
        pd.notna(bbw_pct)
        and pd.notna(vol_z)
        and bbw_pct <= bbw_pct_max
        and vol_z >= volume_z_min
        and spread_bp <= max_spread_bp
    )

    upper_break = max(dc_high.iloc[-1], kc_upper.iloc[-1])
    lower_break = min(dc_low.iloc[-1], kc_lower.iloc[-1])
    metric = 0.0
    if close.iloc[-1] > upper_break:
        metric = (close.iloc[-1] - upper_break) / upper_break
    elif shorting and close.iloc[-1] < lower_break:
        metric = (close.iloc[-1] - lower_break) / lower_break

    side = "none"
    if filters_ok and abs(metric) > score_threshold:
        side = "long" if metric > 0 else "short"

    logger.info(
        "signal=breakout side=%s metric=%.4f vol_z=%.2f bbw_pct=%.2f",
        side,
        metric,
        vol_z if pd.notna(vol_z) else float("nan"),
        bbw_pct if pd.notna(bbw_pct) else float("nan"),
    )
    return metric, side, atr_latest


def should_exit(
    df: pd.DataFrame,
    entry_price: float,
    side: str,
    bars_held: int,
    config: Optional[dict] = None,
) -> bool:
    """Check whether to exit a position based on ATR targets or time."""
    if df is None or df.empty:
        return False
    cfg_all = config or {}
    cfg = cfg_all.get("breakout", cfg_all)
    atr_len = int(cfg.get("atr_len", 14))
    atr_mult_tp = float(cfg.get("atr_mult_tp", 2.0))
    atr_mult_sl = float(cfg.get("atr_mult_sl", 1.0))
    time_exit_bars = int(cfg.get("time_exit_bars", 0))

    close = float(df["close"].iloc[-1])
    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_len)
    atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0

    exit_reason = None
    if side == "long":
        if close >= entry_price + atr_val * atr_mult_tp:
            exit_reason = "tp"
        elif close <= entry_price - atr_val * atr_mult_sl:
            exit_reason = "sl"
    elif side == "short":
        if close <= entry_price - atr_val * atr_mult_tp:
            exit_reason = "tp"
        elif close >= entry_price + atr_val * atr_mult_sl:
            exit_reason = "sl"

    if exit_reason is None and time_exit_bars > 0 and bars_held >= time_exit_bars:
        exit_reason = "time"

    if exit_reason:
        logger.info("signal=exit side=%s reason=%s", side, exit_reason)
        return True
    return False
