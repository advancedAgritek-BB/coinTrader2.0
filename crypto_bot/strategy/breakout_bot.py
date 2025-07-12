from typing import Optional, Tuple
import os
import requests

import pandas as pd
import ta
try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
    if not hasattr(scipy_stats, "norm"):
        raise ImportError
except Exception:  # pragma: no cover - fallback
    class _Norm:
        @staticmethod
        def ppf(_x):
            return 0.0

    class _FakeStats:
        norm = _Norm()

    scipy_stats = _FakeStats()

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils import stats


NEWS_SENTIMENT_URL = os.getenv("NEWS_SENTIMENT_URL", "")


def fetch_news_sentiment(query: str = "crypto") -> int:
    """Return a simple sentiment score from a news page (0-100)."""
    mock = os.getenv("MOCK_NEWS_SENTIMENT")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50
    if not NEWS_SENTIMENT_URL:
        return 50
    try:
        resp = requests.get(f"{NEWS_SENTIMENT_URL}?q={query}", timeout=3)
        resp.raise_for_status()
        text = resp.text.lower()
        pos = sum(w in text for w in ["bull", "surge", "rally", "gain"])
        neg = sum(w in text for w in ["bear", "drop", "slump", "loss"])
        if pos > neg:
            return 70
        if neg > pos:
            return 30
    except Exception:
        return 50
    return 50


def news_sentiment_ok(direction: str) -> bool:
    score = fetch_news_sentiment()
    return score > 55 if direction == "long" else score < 45


def _squeeze(
    df: pd.DataFrame,
    bb_len: int,
    bb_std: float,
    kc_len: int,
    kc_mult: float,
    threshold: float,
    lookback: int,
    squeeze_pct: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return squeeze boolean series and ATR values."""
    hist = max(bb_len, kc_len)
    recent = df.iloc[-(hist + 1) :]

    close = recent["close"]
    high = recent["high"]
    low = recent["low"]

    bb = ta.volatility.BollingerBands(close, window=bb_len, window_dev=bb_std)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    atr = ta.volatility.average_true_range(high, low, close, window=kc_len)
    kc_width = 2 * atr * kc_mult

    if len(bb_width) >= lookback:
        width_z = stats.zscore(bb_width, lookback)
        thresh = scipy_stats.norm.ppf(squeeze_pct / 100)
        squeeze = (width_z < thresh) & (bb_width < kc_width)
    else:
        squeeze = (bb_width / bb_mid < threshold) & (bb_width < kc_width)

    bb_width = cache_series("bb_width", df, bb_width, hist)
    bb_mid = cache_series("bb_mid", df, bb_mid, hist)
    atr = cache_series("atr_kc", df, atr, hist)
    kc_width = cache_series("kc_width", df, kc_width, hist)
    squeeze = cache_series("squeeze", df, squeeze.astype(float), hist) > 0

    return squeeze, atr


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    higher_df: Optional[pd.DataFrame] = None,
    other_df: Optional[pd.DataFrame] = None,
    *,
    _check_other: bool = True,
) -> Tuple[float, str] | Tuple[float, str, float]:
    """Breakout strategy using Bollinger/Keltner squeeze confirmation.

    Returns
    -------
    Tuple[float, str] or Tuple[float, str, float]
        If ``higher_df`` is provided the function returns ``(score, direction)``.
        Otherwise it returns ``(score, direction, atr)`` where ``atr`` is the
        most recent Average True Range value. When ``other_df`` is supplied a
        signal is only returned if the breakout does not appear in ``other_df``.
    """
    if df is None or df.empty:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    cfg_all = config or {}
    cfg = cfg_all.get("breakout", {})
    bb_len = int(cfg.get("bb_length", 20))
    bb_std = float(cfg.get("bb_std", 2))
    kc_len = int(cfg.get("kc_length", 20))
    kc_mult = float(cfg.get("kc_mult", 1.5))
    dc_len = int(cfg.get("dc_length", 10))
    atr_buffer_mult = float(cfg.get("atr_buffer_mult", 0.1))
    vol_window = int(cfg.get("volume_window", 10))
    volume_mult = float(cfg.get("volume_mult", 2))
    exit_drop_pct = float(cfg.get("pump_dump_exit_drop_pct", 0))
    threshold = float(cfg.get("squeeze_threshold", 0.03))
    momentum_filter = bool(cfg.get("momentum_filter", False))
    lookback_cfg = int(cfg_all.get("indicator_lookback", 250))
    squeeze_pct = float(cfg_all.get("bb_squeeze_pct", 20))

    lookback = max(bb_len, kc_len, dc_len, vol_window, 14)
    if len(df) < lookback:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    recent = df.iloc[-(lookback + 1) :]

    squeeze, atr = _squeeze(
        recent,
        bb_len,
        bb_std,
        kc_len,
        kc_mult,
        threshold,
        lookback_cfg,
        squeeze_pct,
    )
    if pd.isna(squeeze.iloc[-1]) or not squeeze.iloc[-1]:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    if higher_df is not None and not higher_df.empty:
        h_sq, _ = _squeeze(
            higher_df.iloc[-(lookback + 1) :],
            bb_len,
            bb_std,
            kc_len,
            kc_mult,
            threshold,
            lookback_cfg,
            squeeze_pct,
        )
        if pd.isna(h_sq.iloc[-1]) or not h_sq.iloc[-1]:
            return 0.0, "none"

    close = recent["close"]
    high = recent["high"]
    low = recent["low"]
    volume = recent["volume"]

    dc_high = high.rolling(dc_len).max().shift(1)
    dc_low = low.rolling(dc_len).min().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    rsi = ta.momentum.rsi(close, window=14)
    macd_hist = ta.trend.macd_diff(close)

    dc_high = cache_series("dc_high", df, dc_high, lookback)
    dc_low = cache_series("dc_low", df, dc_low, lookback)
    vol_ma = cache_series("vol_ma_breakout", df, vol_ma, lookback)
    rsi = cache_series("rsi_breakout", df, rsi, lookback)
    macd_hist = cache_series("macd_hist", df, macd_hist, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["rsi"] = rsi
    recent["macd_hist"] = macd_hist

    latest_vals = [
        dc_high.iloc[-1],
        dc_low.iloc[-1],
        vol_ma.iloc[-1],
        atr.iloc[-1],
        rsi.iloc[-1],
        macd_hist.iloc[-1],
    ]
    if any(pd.isna(v) for v in latest_vals):
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    vol_ma_last = vol_ma.iloc[-1]
    if vol_ma_last <= 0 or pd.isna(vol_ma_last):
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    vol_ok = volume.iloc[-1] > vol_ma_last * volume_mult
    atr_last = atr.iloc[-1]
    upper_break = dc_high.iloc[-1] + atr_last * atr_buffer_mult
    lower_break = dc_low.iloc[-1] - atr_last * atr_buffer_mult

    long_cond = close.iloc[-1] > upper_break
    short_cond = close.iloc[-1] < lower_break

    if momentum_filter:
        long_cond = long_cond and (rsi.iloc[-1] > 60 or macd_hist.iloc[-1] > 0) and news_sentiment_ok("long")
        short_cond = short_cond and (rsi.iloc[-1] < 40 or macd_hist.iloc[-1] < 0) and news_sentiment_ok("short")

    direction = "none"
    score = 0.0
    if long_cond and vol_ok:
        direction = "long"
        score = 1.0
    elif short_cond and vol_ok:
        direction = "short"
        score = 1.0

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(recent, score)

    if higher_df is not None:
        return score, direction

    if score == 0 and exit_drop_pct and len(recent) >= 2:
        prev_vol = volume.iloc[-2]
        prev_vol_ma = vol_ma.iloc[-2]
        prev_upper = dc_high.iloc[-2] + atr.iloc[-2] * atr_buffer_mult
        prev_lower = dc_low.iloc[-2] - atr.iloc[-2] * atr_buffer_mult
        prev_long = (
            close.iloc[-2] > prev_upper and prev_vol > prev_vol_ma * volume_mult
        )
        prev_short = (
            close.iloc[-2] < prev_lower and prev_vol > prev_vol_ma * volume_mult
        )
        if (prev_long or prev_short) and volume.iloc[-1] < prev_vol * exit_drop_pct:
            return (0.0, "none") if higher_df is not None else (0.0, "none", atr_last)

    if _check_other and other_df is not None and not other_df.empty and direction != "none":
        other_score, other_dir, _ = generate_signal(
            other_df,
            config,
            None,
            None,
            _check_other=False,
        )
        if other_dir != "none":
            return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)
        direction = f"arb_{direction}"

    tp_pct = float(cfg.get("trailing_tp_pct", 0.005))
    if direction == "long":
        tp_price = close.iloc[-1] * (1 + tp_pct)
    elif direction == "short":
        tp_price = close.iloc[-1] * (1 - tp_pct)
    else:
        tp_price = close.iloc[-1]

    return score, direction, atr_last


def generate_micro_breakout(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return micro breakout signal based on tick data."""
    if df is None or df.empty:
        return 0.0, "none"

    cfg = config.get("micro_breakout", {}) if config else {}
    dc_len = int(cfg.get("dc_length", 5))
    vol_window = int(cfg.get("volume_window", 5))
    breakout_pct = float(cfg.get("breakout_pct", 0.001))
    volume_mult = float(cfg.get("volume_mult", 1.5))

    lookback = max(dc_len, vol_window)
    if len(df) < lookback + 1:
        return 0.0, "none"

    price_col = "close" if "close" in df.columns else "price"
    recent = df.iloc[-(lookback + 1) :]
    price = recent[price_col]
    volume = recent["volume"]

    high = price.rolling(dc_len).max().shift(1)
    low = price.rolling(dc_len).min().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    if any(pd.isna(x.iloc[-1]) for x in [high, low, vol_ma]):
        return 0.0, "none"
    if vol_ma.iloc[-1] <= 0:
        return 0.0, "none"

    vol_ok = volume.iloc[-1] > vol_ma.iloc[-1] * volume_mult
    upper = high.iloc[-1] * (1 + breakout_pct)
    lower = low.iloc[-1] * (1 - breakout_pct)

    direction = "none"
    if price.iloc[-1] > upper and vol_ok:
        direction = "long"
    elif price.iloc[-1] < lower and vol_ok:
        direction = "short"

    score = 1.0 if direction != "none" else 0.0
    if score and (config is None or config.get("atr_normalization", True)):
        temp = recent[[price_col]].copy()
        temp.rename(columns={price_col: "close"}, inplace=True)
        score = normalize_score_by_volatility(temp, score)

    return score, direction
