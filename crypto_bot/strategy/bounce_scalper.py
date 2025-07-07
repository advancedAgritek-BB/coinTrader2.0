from dataclasses import dataclass, fields
from typing import Optional, Tuple, Union

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


@dataclass
class BounceScalperConfig:
    """Configuration options for :func:`generate_signal`."""

    rsi_window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    vol_window: int = 20
    volume_multiple: float = 2.0
    zscore_threshold: float = 1.5
    down_candles: int = 3
    up_candles: int = 3
    trend_ema_fast: int = 9
    trend_ema_slow: int = 21
    cooldown_bars: int = 2
    atr_period: int = 14
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.0
    min_score: float = 0.3
    max_concurrent_signals: int = 1
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "BounceScalperConfig":
        """Build a config from a dictionary."""
        cfg = cfg or {}
        params = {f.name: cfg.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        return cls(**params)


def generate_signal(
    df: pd.DataFrame, config: Optional[Union[dict, BounceScalperConfig]] = None
) -> Tuple[float, str]:
    """Identify short-term bounces with volume confirmation."""
    if df.empty:
        return 0.0, "none"

    cfg = config or {}
    rsi_window = int(cfg.get("rsi_window", 14))
    oversold = float(cfg.get("oversold", 30))
    overbought = float(cfg.get("overbought", 70))
    vol_window = int(cfg.get("vol_window", 20))
    volume_multiple = float(cfg.get("volume_multiple", 2.0))
    zscore_threshold = float(cfg.get("zscore_threshold", 2.0))
    down_candles = int(cfg.get("down_candles", 3))
    params = (
        config
        if isinstance(config, BounceScalperConfig)
        else BounceScalperConfig.from_dict(config)
    )

    rsi_window = params.rsi_window
    oversold = params.oversold
    overbought = params.overbought
    vol_window = params.vol_window
    volume_multiple = params.volume_multiple
    down_candles = params.down_candles

    lookback = max(rsi_window, vol_window, down_candles + 1)
    if len(df) < lookback:
        return 0.0, "none"

    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_window)
    df["vol_ma"] = df["volume"].rolling(window=vol_window).mean()
    df["vol_std"] = df["volume"].rolling(window=vol_window).std()
    recent = df.iloc[-(lookback + 1) :]

    rsi = ta.momentum.rsi(recent["close"], window=rsi_window)
    vol_ma = recent["volume"].rolling(window=vol_window).mean()

    rsi = cache_series("rsi", df, rsi, lookback)
    vol_ma = cache_series("vol_ma", df, vol_ma, lookback)

    df = recent.copy()
    df["rsi"] = rsi
    df["vol_ma"] = vol_ma

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    vol_z = (latest["volume"] - latest["vol_ma"]) / latest["vol_std"] if latest["vol_std"] > 0 else float("inf")
    volume_spike = vol_z > zscore_threshold

    recent_changes = df["close"].diff()
    downs = (recent_changes.iloc[-down_candles - 1 : -1] < 0).all()
    ups = (recent_changes.iloc[-down_candles - 1 : -1] > 0).all()

    score = 0.0
    direction = "none"

    if (
        not pd.isna(latest["rsi"])
        and downs
        and latest["close"] > prev_close
        and latest["rsi"] < oversold
        and volume_spike
    ):
        rsi_score = min((oversold - latest["rsi"]) / oversold, 1.0)
        vol_score = min(latest["volume"] / (latest["vol_ma"] * volume_multiple), 1.0)
        score = (rsi_score + vol_score) / 2
        direction = "long"
    elif (
        not pd.isna(latest["rsi"])
        and ups
        and latest["close"] < prev_close
        and latest["rsi"] > overbought
        and volume_spike
    ):
        rsi_score = min((latest["rsi"] - overbought) / (100 - overbought), 1.0)
        vol_score = min(latest["volume"] / (latest["vol_ma"] * volume_multiple), 1.0)
        score = (rsi_score + vol_score) / 2
        direction = "short"

    if score > 0 and params.atr_normalization:
        score = normalize_score_by_volatility(df, score)

    return score, direction
