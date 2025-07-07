from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple, Union

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.cooldown_manager import in_cooldown, mark_cooldown


@dataclass
class BounceScalperConfig:
    """Configuration for :func:`generate_signal`."""

    rsi_window: int = 14
    oversold: float = 30
    overbought: float = 70
    vol_window: int = 20
    zscore_threshold: float = 2.0
    volume_multiple: float = 2.0
    ema_window: int = 50
    atr_window: int = 14
    down_candles: int = 3
    strategy: str = "bounce_scalper"
    symbol: str = ""
    atr_normalization: bool = True

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


def generate_signal(df: pd.DataFrame, config: ConfigType = None) -> Tuple[float, str]:
    """Identify short-term bounces with volume confirmation."""
    if df.empty:
        return 0.0, "none"

    cfg = _as_dict(config)

    symbol = cfg.get("symbol", "")
    strategy = cfg.get("strategy", "bounce_scalper")
    if symbol and in_cooldown(symbol, strategy):
        return 0.0, "none"

    rsi_window = int(cfg.get("rsi_window", 14))
    oversold = float(cfg.get("oversold", 30))
    overbought = float(cfg.get("overbought", 70))
    vol_window = int(cfg.get("vol_window", 20))
    zscore_threshold = float(cfg.get("zscore_threshold", 2.0))
    volume_multiple = float(cfg.get("volume_multiple", 2.0))
    ema_window = int(cfg.get("ema_window", 50))
    atr_window = int(cfg.get("atr_window", 14))
    down_candles = int(cfg.get("down_candles", 3))

    lookback = max(rsi_window, vol_window, ema_window, atr_window, down_candles + 1)
    if len(df) < lookback:
        return 0.0, "none"

    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_window)
    df["vol_ma"] = df["volume"].rolling(window=vol_window).mean()
    df["vol_std"] = df["volume"].rolling(window=vol_window).std()
    df["ema"] = ta.trend.ema_indicator(df["close"], window=ema_window)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window
    )

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    if not pd.isna(latest["vol_std"]) and latest["vol_std"] > 0:
        zscore = (latest["volume"] - latest["vol_ma"]) / latest["vol_std"]
        volume_spike = zscore > zscore_threshold
    else:
        volume_spike = (
            latest["volume"] > latest["vol_ma"] * volume_multiple
            if latest["vol_ma"] > 0
            else False
        )

    if not pd.isna(latest["atr"]) and latest["close"] > 0:
        atr_pct = latest["atr"] / latest["close"] * 100
        oversold = max(10.0, oversold - atr_pct * 0.5)
        overbought = min(90.0, overbought + atr_pct * 0.5)

    trend_ok_long = latest["close"] > latest["ema"]
    trend_ok_short = latest["close"] < latest["ema"]

    recent_changes = df["close"].diff()
    downs = (recent_changes.iloc[-down_candles - 1 : -1] < 0).all()
    ups = (recent_changes.iloc[-down_candles - 1 : -1] > 0).all()

    score = 0.0
    direction = "none"

    if (
        not pd.isna(latest["rsi"])
        and downs
        and latest["close"] > prev_close
        and trend_ok_long
        and latest["rsi"] < oversold
        and volume_spike
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
        and latest["rsi"] > overbought
        and volume_spike
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
        if cfg.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        if symbol:
            mark_cooldown(symbol, strategy)

    return score, direction
