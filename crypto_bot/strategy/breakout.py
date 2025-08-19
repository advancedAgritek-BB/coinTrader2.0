from __future__ import annotations

from typing import Optional, Tuple

import logging
import pandas as pd
import ta

from crypto_bot.utils import stats
from crypto_bot.utils.indicator_cache import cache_series

NAME = "breakout"
logger = logging.getLogger(__name__)


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
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
    if df is None or df.empty:
        logger.info(
            "signal=breakout side=none reason=insufficient_data vol_z=nan bbw_pct=nan"
        )
        return 0.0, "none", 0.0

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
    allow_short = bool(cfg.get("allow_short", False))

    lookback = max(donchian_len, keltner_len, bbw_lookback, vol_window, atr_len)
    if len(df) < lookback + 1:
        logger.info(
            "signal=breakout side=none reason=insufficient_data vol_z=nan bbw_pct=nan"
        )
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

    long_reason = None
    short_reason = None
    if (
        close.iloc[-1] > dc_high.iloc[-1]
        and close.iloc[-2] <= dc_high.iloc[-2]
    ):
        long_reason = "donchian"
    elif close.iloc[-1] > kc_upper.iloc[-1]:
        long_reason = "keltner"

    if allow_short:
        if (
            close.iloc[-1] < dc_low.iloc[-1]
            and close.iloc[-2] >= dc_low.iloc[-2]
        ):
            short_reason = "donchian"
        elif close.iloc[-1] < kc_lower.iloc[-1]:
            short_reason = "keltner"

    side = "none"
    score = 0.0
    reason = "none"
    if filters_ok:
        if long_reason:
            side = "long"
            score = 1.0
            reason = long_reason
        elif short_reason:
            side = "short"
            score = 1.0
            reason = short_reason

    logger.info(
        "signal=breakout side=%s reason=%s vol_z=%.2f bbw_pct=%.2f",
        side,
        reason,
        vol_z if pd.notna(vol_z) else float("nan"),
        bbw_pct if pd.notna(bbw_pct) else float("nan"),
    )
    return score, side, atr_latest


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
