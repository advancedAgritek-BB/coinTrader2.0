from pathlib import Path
from typing import Optional, Tuple, Dict
import asyncio
import time
from .pattern_detector import detect_patterns

import pandas as pd
import numpy as np
import ta
import yaml
from crypto_bot.utils.logger import setup_logger


CONFIG_PATH = Path(__file__).with_name("regime_config.yaml")


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config(CONFIG_PATH)

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def _ml_fallback(df: pd.DataFrame) -> Tuple[str, float]:
    """Return regime label and confidence using the bundled ML model."""
    try:
        from .ml_fallback import predict_regime

        label, confidence = predict_regime(df)
        return label, confidence
    except Exception:
        return "unknown", 0.0


def _classify_core(
    data: pd.DataFrame, cfg: dict, higher_df: Optional[pd.DataFrame] = None
) -> str:
    if data is None or data.empty or len(data) < 20:
        return "unknown"

    df = data.copy()
    for col in ("ema20", "ema50", "adx", "rsi", "atr", "bb_width"):
        df[col] = np.nan

    if len(df) >= cfg["ema_fast"]:
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=cfg["ema_fast"])

    if len(df) >= cfg["ema_slow"]:
        df["ema50"] = ta.trend.ema_indicator(df["close"], window=cfg["ema_slow"])

    if len(df) >= cfg["indicator_window"]:
        try:
            df["adx"] = ta.trend.adx(
                df["high"], df["low"], df["close"], window=cfg["indicator_window"]
            )
            df["rsi"] = ta.momentum.rsi(df["close"], window=cfg["indicator_window"])
            df["atr"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=cfg["indicator_window"]
            )
        except IndexError:
            return "unknown"

    if len(df) >= cfg["bb_window"]:
        bb = ta.volatility.BollingerBands(df["close"], window=cfg["bb_window"])
        df["bb_width"] = bb.bollinger_wband()

    volume_ma20 = (
        df["volume"].rolling(cfg["ma_window"]).mean()
        if len(df) >= cfg["ma_window"]
        else pd.Series(np.nan, index=df.index)
    )
    atr_ma20 = (
        df["atr"].rolling(cfg["ma_window"]).mean()
        if len(df) >= cfg["ma_window"]
        else pd.Series(np.nan, index=df.index)
    )

    latest = df.iloc[-1]

    trending = latest["adx"] > cfg["adx_trending_min"] and latest["ema20"] > latest["ema50"]

    if trending and cfg.get("confirm_trend_with_higher_tf", False):
        if higher_df is None:
            trending = False
        else:
            confirm_cfg = cfg.copy()
            confirm_cfg["confirm_trend_with_higher_tf"] = False
            if _classify_core(higher_df, confirm_cfg, None) != "trending":
                trending = False

    regime = "sideways"

    if trending:
        regime = "trending"
    elif (
        latest["adx"] < cfg["adx_sideways_max"]
        and latest["bb_width"] < cfg["bb_width_sideways_max"]
    ):
        regime = "sideways"
    elif (
        latest["bb_width"] < cfg["bb_width_breakout_max"]
        and not np.isnan(volume_ma20.iloc[-1])
        and latest["volume"] > volume_ma20.iloc[-1] * cfg["breakout_volume_mult"]
    ):
        regime = "breakout"
    elif (
        cfg["rsi_mean_rev_min"] <= latest["rsi"] <= cfg["rsi_mean_rev_max"]
        and abs(latest["close"] - latest["ema20"]) / latest["close"]
        < cfg["ema_distance_mean_rev_max"]
    ):
        regime = "mean-reverting"
    elif (
        not np.isnan(atr_ma20.iloc[-1])
        and latest["atr"] > atr_ma20.iloc[-1] * cfg["atr_volatility_mult"]
    ):
        regime = "volatile"

    return regime


def classify_regime(
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    config_path: Optional[str] = None,
) -> Tuple[str, object]:
    """Classify market regime.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    config_path : Optional[str], default None
        Optional path to override the default configuration. Primarily used for
        testing.

    Returns
    -------
    Tuple[str, object]
        When sufficient history is available the function returns ``(label,
        patterns)`` where ``patterns`` is a ``set`` of detected formations. If
        insufficient history triggers the ML fallback the return value is
        ``(label, confidence)`` where ``confidence`` is a float between 0 and 1.
    """

    cfg = CONFIG if config_path is None else _load_config(Path(config_path))
    ml_min_bars = cfg.get("ml_min_bars", 20)

    regime = _classify_core(df, cfg, higher_df)

    if regime == "unknown" and cfg.get("use_ml_regime_classifier", False):
        if len(df) >= ml_min_bars:
            try:  # pragma: no cover - safety net
                from .ml_regime_model import predict_regime

                regime = predict_regime(df)
            except Exception:
                pass
        else:
            logger.info(
                "Skipping ML fallback \u2014 insufficient data (%d rows)", len(df)
            )

    patterns = detect_patterns(df)
    if "breakout" in patterns:
        regime = "breakout"

    if regime == "unknown":
        if len(df) >= ml_min_bars:
            label, confidence = _ml_fallback(df)
            return label, confidence
        logger.info("Skipping ML fallback \u2014 insufficient data (%d rows)", len(df))
        return regime, patterns

    return regime, patterns


async def classify_regime_async(
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    config_path: Optional[str] = None,
) -> Tuple[str, object]:
    """Asynchronous wrapper around :func:`classify_regime`."""
    return await asyncio.to_thread(
        classify_regime, df, higher_df, config_path=config_path
    )


# Caching utilities -----------------------------------------------------

regime_cache: Dict[tuple[str, str], str] = {}
_regime_cache_ts: Dict[tuple[str, str], int] = {}


async def classify_regime_cached(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    profile: bool = False,
    *,
    config_path: Optional[str] = None,
) -> Tuple[str, object]:
    """Classify ``symbol`` regime with caching and optional profiling."""

    if df is None or df.empty:
        return "unknown", 0.0

    ts = int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else len(df)
    key = (symbol, timeframe)
    if key in regime_cache and _regime_cache_ts.get(key) == ts:
        label = regime_cache[key]
        # Info is not cached; recompute minimal patterns for compatibility
        return label, set()

    start = time.perf_counter() if profile else 0.0
    label, info = await classify_regime_async(df, higher_df, config_path=config_path)
    regime_cache[key] = label
    _regime_cache_ts[key] = ts
    if profile:
        logger.info(
            "Regime classification for %s %s took %.4fs",
            symbol,
            timeframe,
            time.perf_counter() - start,
        )
    return label, info


def clear_regime_cache(symbol: str, timeframe: str) -> None:
    """Remove cached regime entry for ``symbol`` and ``timeframe``."""
    regime_cache.pop((symbol, timeframe), None)
    _regime_cache_ts.pop((symbol, timeframe), None)


