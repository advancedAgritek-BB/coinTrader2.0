"""Short term bounce detection strategy."""

from dataclasses import asdict, dataclass, fields
from typing import Callable, Optional, Tuple, Union

from crypto_bot import cooldown_manager

import logging
import pandas as pd
import numpy as np
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
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils import stats

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.cooldown_manager import in_cooldown, mark_cooldown
from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("bounce_scalper")
else:  # pragma: no cover - fallback
    MODEL = None


@dataclass
class BounceScalperConfig:
    """Configuration options for :func:`generate_signal`."""

    # core indicators
    rsi_window: int = 14
    oversold: float = 30.0
    overbought: float = 65.0
    vol_window: int = 20
    vol_zscore_threshold: float = 1.0
    zscore_threshold: float = 1.5
    volume_multiple: float = 1.5
    ema_window: int = 50
    atr_window: int = 14
    lookback: int = 250
    rsi_overbought_pct: float = 90.0
    rsi_oversold_pct: float = 10.0

    # pattern confirmation
    down_candles: int = 2
    up_candles: int = 3
    trend_ema_fast: int = 9
    trend_ema_slow: int = 21

    # risk management
    atr_period: int = 14
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.0
    min_score: float = 0.3
    max_concurrent_signals: int = 1
    atr_normalization: bool = True
    cooldown_enabled: bool = False

    # pattern detection
    pattern_timeframe: str = "1m"

    # metadata
    strategy: str = "bounce_scalper"
    symbol: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "BounceScalperConfig":
        """Create config from a dictionary."""
        params = {}
        for f in fields(cls):
            params[f.name] = data.get(f.name, f.default)
        return cls(**params)


ConfigType = Union[dict, BounceScalperConfig, None]


def _as_dict(cfg: ConfigType) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, BounceScalperConfig):
        return asdict(cfg)
    return dict(cfg)


def is_engulfing(df: pd.DataFrame, body_pct: float) -> Optional[str]:
    """Return ``"bullish"`` or ``"bearish"`` if the last candle engulfs the
    previous one and its body is at least ``body_pct`` of the range."""
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    last = df.iloc[-1]

    rng = last["high"] - last["low"]
    if rng <= 0:
        return None

    body_ratio = abs(last["close"] - last["open"]) / rng
    if body_ratio < body_pct:
        return None

    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["open"] <= prev["close"]
        and last["close"] >= prev["open"]
    ):
        return "bullish"

    if (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["open"] >= prev["close"]
        and last["close"] <= prev["open"]
    ):
        return "bearish"

    return None


def is_hammer(df: pd.DataFrame, body_pct: float) -> Optional[str]:
    """Return ``"bullish"`` or ``"bearish"`` if the last candle is a hammer
    with body at least ``body_pct`` of its range."""
    if df.empty:
        return None

    last = df.iloc[-1]
    rng = last["high"] - last["low"]
    if rng <= 0:
        return None

    body = abs(last["close"] - last["open"])
    body_ratio = body / rng
    if body_ratio < body_pct:
        return None

    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    if lower > upper * 2:
        return "bullish"
    if upper > lower * 2:
        return "bearish"

    return None


def confirm_higher_lows(df: pd.DataFrame, bars: int) -> bool:
    """Return ``True`` if the last ``bars`` lows are strictly increasing."""
    if bars <= 1 or len(df) < bars:
        return True

    lows = df["low"].iloc[-bars:]
    return lows.diff().dropna().gt(0).all()


def confirm_lower_highs(df: pd.DataFrame, bars: int) -> bool:
    """Return ``True`` if the last ``bars`` highs are strictly decreasing."""
    if bars <= 1 or len(df) < bars:
        return True

    highs = df["high"].iloc[-bars:]
    return highs.diff().dropna().lt(0).all()


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Union[dict, BounceScalperConfig]] = None,
    *,
    lower_df: Optional[pd.DataFrame] = None,
    fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    book: Optional[dict] = None,
    force: bool = False,
) -> Tuple[float, str]:
    """Identify short-term bounces with volume confirmation.

    Setting ``force=True`` bypasses the cooldown and recent win-rate filters
    for a single invocation.
    """
    if df.empty:
        return 0.0, "none"

    cfg_dict = _as_dict(config)
    cfg = BounceScalperConfig.from_dict(cfg_dict)

    symbol = cfg.symbol
    if len(df) < 50:
        return 0.0, "none"

    strategy = cfg.strategy

    if (
        cfg.cooldown_enabled
        and symbol
        and in_cooldown(symbol, strategy)
        and not force
    ):
        if get_recent_win_rate() >= 0.5:
            return 0.0, "none"

    rsi_window = cfg.rsi_window
    oversold = cfg.oversold
    overbought = cfg.overbought
    vol_window = cfg.vol_window
    zscore_threshold = cfg.zscore_threshold
    volume_multiple = cfg.volume_multiple
    ema_window = cfg.ema_window
    atr_window = cfg.atr_window
    down_candles = cfg.down_candles
    up_candles = cfg.up_candles
    body_pct = cfg_dict.get("body_pct", 0.5)

    lookback = min(cfg.lookback, len(df)) if cfg.lookback else len(df)

    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_window)
    df["rsi_z"] = stats.zscore(df["rsi"], cfg.lookback)
    df["vol_ma"] = df["volume"].rolling(window=vol_window).mean()
    df["vol_std"] = df["volume"].rolling(window=vol_window).std()
    df["ema"] = ta.trend.ema_indicator(df["close"], window=ema_window)
    atr_window_used = min(atr_window, len(df))
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window_used
    )

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    if not pd.isna(latest["vol_std"]) and latest["vol_std"] > 0:
        zscore = (latest["volume"] - latest["vol_ma"]) / latest["vol_std"]
    else:
        zscore = float("inf")
    volume_spike = (
        zscore > zscore_threshold
        or (
            latest["volume"] > latest["vol_ma"] * volume_multiple
            if latest["vol_ma"] > 0
            else False
        )
    )

    if not pd.isna(latest["atr"]) and latest["close"] > 0:
        atr_pct = latest["atr"] / latest["close"] * 100
        oversold = max(10.0, oversold - atr_pct * 0.5)
        overbought = min(90.0, overbought + atr_pct * 0.5)

    trend_ok_long = pd.isna(latest["ema"]) or latest["close"] > latest["ema"]
    trend_ok_short = pd.isna(latest["ema"]) or latest["close"] < latest["ema"]

    recent = df.iloc[-(lookback + 1) :]
    rsi_series = ta.momentum.rsi(recent["close"], window=rsi_window)
    vol_ma = recent["volume"].rolling(window=vol_window).mean()
    rsi_series = cache_series("rsi", df, rsi_series, lookback)
    vol_ma = cache_series("vol_ma", df, vol_ma, lookback)

    df = recent.copy()
    df["rsi"] = rsi_series
    df["rsi_z"] = stats.zscore(rsi_series, lookback)
    df["vol_ma"] = vol_ma

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    vol_z = (
        (latest["volume"] - latest["vol_ma"]) / latest["vol_std"]
        if latest["vol_std"] > 0
        else float("inf")
    )
    volume_spike = vol_z > zscore_threshold

    pattern_df = lower_df or df
    if lower_df is None and fetcher and cfg.pattern_timeframe:
        try:
            pattern_df = fetcher(cfg.pattern_timeframe)
        except Exception:  # pragma: no cover - safety
            pass

    eng_type = is_engulfing(pattern_df, body_pct)
    hammer_type = is_hammer(pattern_df, body_pct)
    bull_pattern = eng_type == "bullish" or hammer_type == "bullish"
    bear_pattern = eng_type == "bearish" or hammer_type == "bearish"

    higher_lows = confirm_higher_lows(df, up_candles)
    lower_highs = confirm_lower_highs(df, up_candles)

    recent_changes = df["close"].diff()
    downs = (recent_changes.iloc[-down_candles - 1 : -1] < 0).all()
    ups = (recent_changes.iloc[-down_candles - 1 : -1] > 0).all()

    score = 0.0
    direction = "none"

    rsi_z_last = df["rsi_z"].iloc[-1]
    lower_thresh = scipy_stats.norm.ppf(cfg.rsi_oversold_pct / 100)
    upper_thresh = scipy_stats.norm.ppf(cfg.rsi_overbought_pct / 100)
    oversold_cond = (
        rsi_z_last < lower_thresh
        if not pd.isna(rsi_z_last)
        else latest["rsi"] < oversold
    )
    overbought_cond = (
        rsi_z_last > upper_thresh
        if not pd.isna(rsi_z_last)
        else latest["rsi"] > overbought
    )

    if (
        not pd.isna(latest["rsi"])
        and downs
        and latest["close"] > prev_close
        and trend_ok_long
        and oversold_cond
        and volume_spike
        and bull_pattern
        and higher_lows
    ):
        rsi_score = min((oversold - latest["rsi"]) / oversold, 1.0)
        vol_score = min(
            latest["volume"] / (latest["vol_ma"] * volume_multiple)
            if latest["vol_ma"] > 0
            else 0,
            1.0,
        )
        score = (rsi_score + vol_score) / 2
        direction = "long"
    elif (
        not pd.isna(latest["rsi"])
        and ups
        and latest["close"] < prev_close
        and trend_ok_short
        and overbought_cond
        and volume_spike
        and bear_pattern
        and lower_highs
    ):
        rsi_score = min((latest["rsi"] - overbought) / (100 - overbought), 1.0)
        vol_score = min(
            latest["volume"] / (latest["vol_ma"] * volume_multiple)
            if latest["vol_ma"] > 0
            else 0,
            1.0,
        )
        score = (rsi_score + vol_score) / 2
        direction = "short"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        if symbol:
            mark_cooldown(symbol, strategy)
        book_data = book or cfg_dict.get("order_book")
        ratio = float(cfg_dict.get("imbalance_ratio", 0))
        penalty = float(cfg_dict.get("imbalance_penalty", 0))
        if ratio and isinstance(book_data, dict):
            bids = sum(v for _, v in book_data.get("bids", []))
            asks = sum(v for _, v in book_data.get("asks", []))
            if bids and asks:
                imbalance = bids / asks
                if direction == "long" and imbalance < ratio:
                    if penalty:
                        score *= max(0.0, 1 - penalty)
                    else:
                        return 0.0, "none"
                if direction == "short" and imbalance > ratio:
                    if penalty:
                        score *= max(0.0, 1 - penalty)
                    else:
                        return 0.0, "none"

    return score, direction


class regime_filter:
    """Return True for the bounce regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "bounce"
