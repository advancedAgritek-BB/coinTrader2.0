from pathlib import Path
from typing import Optional, Tuple, Dict
import asyncio
import time
from .pattern_detector import detect_patterns
from crypto_bot.utils.pattern_logger import log_patterns

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

# Impact of each detected pattern on regime scoring. Values are multipliers
# applied to the pattern strength.
PATTERN_WEIGHTS = {
    "breakout": ("breakout", 2.0),
    "breakdown": ("volatile", 1.0),
    "hammer": ("mean-reverting", 0.5),
    "shooting_star": ("mean-reverting", 0.5),
    "doji": ("sideways", 0.2),
    "ascending_triangle": ("breakout", 1.5),
}


def _ml_fallback(df: pd.DataFrame) -> Tuple[str, float]:
    """Return regime label and confidence using the bundled gradient boosting model."""
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
            df["normalized_range"] = (df["high"] - df["low"]) / df["atr"]
        except IndexError:
            return "unknown"
            df["adx"] = np.nan
            df["rsi"] = np.nan
            df["atr"] = np.nan
            df["normalized_range"] = np.nan
            return "unknown"
    else:
        df["adx"] = np.nan
        df["rsi"] = np.nan
        df["atr"] = np.nan
        df["normalized_range"] = np.nan

    if len(df) >= cfg["bb_window"]:
        bb = ta.volatility.BollingerBands(df["close"], window=cfg["bb_window"])
        df["bb_width"] = bb.bollinger_wband()

    df["volume_change"] = df["volume"].pct_change()
    if len(df) >= cfg["ma_window"]:
        mean_change = df["volume_change"].rolling(cfg["ma_window"]).mean()
        std_change = df["volume_change"].rolling(cfg["ma_window"]).std()
        df["volume_zscore"] = (df["volume_change"] - mean_change) / std_change
    else:
        df["volume_zscore"] = np.nan

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

    volume_jump = False
    if len(df) > 1:
        vol_change = df["volume_change"].iloc[-1]
        vol_z = df["volume_zscore"].iloc[-1]
        if (
            (not np.isnan(vol_change) and vol_change > 1.0)
            or (not np.isnan(vol_z) and vol_z > 3)
        ):
            volume_jump = True

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

    if (
        latest["bb_width"] < cfg["bb_width_breakout_max"]
        and not np.isnan(volume_ma20.iloc[-1])
        and latest["volume"] > volume_ma20.iloc[-1] * cfg["breakout_volume_mult"]
    ) or volume_jump:
        regime = "breakout"
    elif trending:
        regime = "trending"
    elif (
        latest["adx"] < cfg["adx_sideways_max"]
        and latest["bb_width"] < cfg["bb_width_sideways_max"]
    ):
        regime = "sideways"
    elif (
        cfg["rsi_mean_rev_min"] <= latest["rsi"] <= cfg["rsi_mean_rev_max"]
        and abs(latest["close"] - latest["ema20"]) / latest["close"]
        < cfg["ema_distance_mean_rev_max"]
    ):
        regime = "mean-reverting"
    elif (
        not np.isnan(latest["normalized_range"])
        and latest["normalized_range"] > cfg["normalized_range_volatility_min"]
    ):
        regime = "volatile"

    return regime


def classify_regime(
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: Optional[str] = None,
) -> Tuple[str, object] | Dict[str, str] | Tuple[str, str]:
    """Classify market regime.

    Parameters
    ----------
    df : pd.DataFrame | None
        OHLCV data for the base timeframe.
    df_map : dict[str, pd.DataFrame] | None
        Optional mapping of timeframe to dataframes. When provided the function
        returns only the regime labels for each timeframe without pattern
        information.
    config_path : Optional[str], default None
        Optional path to override the default configuration. Primarily used for
        testing.

    Returns
    -------
    Tuple[str, object]
        When sufficient history is available the function returns ``(label,
        pattern_scores)`` where ``pattern_scores`` is a ``dict`` mapping pattern
        names to confidence values.  If insufficient history triggers the ML
        fallback the return value is ``(label, confidence)`` where ``confidence``
        is a float between 0 and 1.
        patterns)`` where ``patterns`` is a mapping of pattern names to
        confidence scores. If insufficient history triggers the ML fallback the
        return value is ``(label, confidence)`` where ``confidence`` is a float
        between 0 and 1.
        patterns)`` where ``patterns`` is a ``dict`` mapping formation name to
        strength. If insufficient history triggers the ML fallback the return
        value is ``(label, confidence)`` where ``confidence`` is a float between
        0 and 1.
    Tuple[str, object] | Dict[str, str] | Tuple[str, str]
        When a single dataframe is supplied the function behaves like before
        and returns ``(label, patterns)`` or ``(label, confidence)``. When a
        mapping of dataframes is provided the regimes for each timeframe are
        returned either as a dictionary or, if exactly two timeframes are
        supplied, as a tuple respecting the insertion order of ``df_map``.
    """

    cfg = CONFIG if config_path is None else _load_config(Path(config_path))
    ml_min_bars = cfg.get("ml_min_bars", 20)

    if df_map is not None:
        labels: Dict[str, str] = {}
        for tf, frame in df_map.items():
            h_df = None
            if tf != cfg.get("higher_timeframe"):
                h_df = df_map.get(cfg.get("higher_timeframe"))
            r, _ = classify_regime(frame, h_df, config_path=config_path)
            labels[tf] = r
        if len(df_map) == 2:
            return tuple(labels[tf] for tf in df_map.keys())  # type: ignore
        return labels

    if df is None:
        raise ValueError("df must be provided when df_map is None")

    regime = _classify_core(df, cfg, higher_df)
    used_ml = False
    ml_used = False

    if regime == "unknown" and cfg.get("use_ml_regime_classifier", False):
        if len(df) >= ml_min_bars:
            try:  # pragma: no cover - safety net
                from .ml_regime_model import predict_regime

                regime = predict_regime(df)
                used_ml = True
                ml_used = True
            except Exception:
                pass
        else:
            logger.info(
                "Skipping ML fallback \u2014 insufficient data (%d rows)", len(df)
            )

    pattern_scores = detect_patterns(df)

    regime_scores: Dict[str, float] = {regime: 1.0}
    if pattern_scores.get("breakout", 0) > 0.8:
        regime_scores["breakout"] = regime_scores.get("breakout", 0) + 0.2

    final_regime = max(regime_scores, key=regime_scores.get)

    if final_regime == "unknown":
        if len(df) >= ml_min_bars:
    patterns = detect_patterns(df)

    if regime == "unknown" and len(df) < ml_min_bars:
        log_patterns(regime, patterns)
        return regime, 0.0

    # Score regimes based on the base classification and detected patterns
    scores: Dict[str, float] = {}
    if regime != "unknown":
        scores[regime] = 1.0
    for name, strength in patterns.items():
        target, weight = PATTERN_WEIGHTS.get(name, (None, 0.0))
        if target:
            if regime == "unknown" and name not in {"breakout", "ascending_triangle"}:
                continue
            scores[target] = scores.get(target, 0.0) + weight * float(strength)

    if scores:
        regime = max(scores, key=scores.get)

    log_patterns(regime, patterns)

    if regime == "unknown":
        if cfg.get("use_ml_regime_classifier", False) and len(df) >= ml_min_bars:
            label, confidence = _ml_fallback(df)
            log_patterns(label, patterns)
            return label, confidence
        if len(df) >= ml_min_bars:
            logger.info("Skipping ML fallback \u2014 ML disabled")
        else:
            logger.info("Skipping ML fallback \u2014 insufficient data (%d rows)", len(df))
        return regime, {}
    patterns = set()
    if not ml_used:
        patterns = detect_patterns(df)
        if "breakout" in patterns:
            regime = "breakout"

    if regime == "unknown":
        if len(df) < ml_min_bars:
            return regime, 0.0
        logger.info("Skipping ML fallback \u2014 insufficient data (%d rows)", len(df))
        return final_regime, pattern_scores

    return final_regime, pattern_scores
    return regime, {} if used_ml else patterns


async def classify_regime_async(
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: Optional[str] = None,
) -> Tuple[str, object] | Dict[str, str] | Tuple[str, str]:
    """Asynchronous wrapper around :func:`classify_regime`."""
    return await asyncio.to_thread(
        classify_regime,
        df,
        higher_df,
        df_map=df_map,
        config_path=config_path,
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


