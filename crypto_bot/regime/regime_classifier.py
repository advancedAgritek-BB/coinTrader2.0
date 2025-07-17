from pathlib import Path
from typing import Dict, Optional, Tuple
import asyncio
import time
import logging

import pandas as pd
import numpy as np
import ta
import yaml

from .pattern_detector import detect_patterns
from crypto_bot.utils.pattern_logger import log_patterns
from crypto_bot.utils.logger import LOG_DIR, setup_logger


CONFIG_PATH = Path(__file__).with_name("regime_config.yaml")


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config(CONFIG_PATH)


logger = setup_logger(__name__, LOG_DIR / "bot.log")


def _configure_logger(cfg: dict) -> None:
    level_str = str(cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)


_configure_logger(CONFIG)

_ALL_REGIMES = [
    "trending",
    "sideways",
    "mean-reverting",
    "breakout",
    "volatile",
    "bullish_trending",
    "bearish_volatile",
    "unknown",
]

# Impact of each detected pattern on regime scoring. Values are multipliers
# applied to the pattern strength.
PATTERN_WEIGHTS = {
    "breakout": ("breakout", 2.0),
    "breakdown": ("volatile", 1.0),
    "hammer": ("mean-reverting", 0.5),
    "shooting_star": ("mean-reverting", 0.5),
    "doji": ("sideways", 0.2),
    "bullish_engulfing": ("mean-reverting", 1.2),
    "ascending_triangle": ("breakout", 2.0),
}


def adaptive_thresholds(cfg: dict, df: pd.DataFrame | None, symbol: str | None) -> dict:
    """Return a copy of ``cfg`` with thresholds scaled based on volatility.

    The average ATR over ``df`` is compared to ``cfg["atr_baseline"]`` and
    multipliers are applied to selected thresholds. When ``statsmodels`` is
    available an Augmented Dickey-Fuller and simple autoregression test are used
    to detect drift. When drift is present the RSI limits are widened slightly
    to reduce false mean-reversion signals.
    """

    if df is None or df.empty:
        return cfg

    out = cfg.copy()
    baseline = cfg.get("atr_baseline")
    if baseline:
        try:
            atr = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=cfg.get("indicator_window", 14)
            )
            avg_atr = float(atr.mean())
            factor = avg_atr / float(baseline) if baseline else 1.0
            out["adx_trending_min"] = cfg["adx_trending_min"] * factor
            out["normalized_range_volatility_min"] = cfg[
                "normalized_range_volatility_min"
            ] * factor
        except Exception:
            pass

    try:  # pragma: no cover - optional dependency
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.ar_model import AutoReg

        close = df["close"].dropna()
        if len(close) >= 20:
            pval = adfuller(close, regression="ct")[1]
            ar_res = AutoReg(close, lags=1, old_names=False).fit()
            slope = float(ar_res.params.get("close.L1", 0.0))
            if pval > 0.1 or abs(slope) > 0.9:
                adj = 5
                out["rsi_mean_rev_min"] = max(0, cfg["rsi_mean_rev_min"] - adj)
                out["rsi_mean_rev_max"] = min(100, cfg["rsi_mean_rev_max"] + adj)
    except Exception:
        pass

    return out


def _ml_fallback(df: pd.DataFrame) -> Tuple[str, float]:
    """Return regime label and confidence using the bundled ML fallback model."""
    try:  # pragma: no cover - optional dependency
        from .ml_fallback import predict_regime
    except Exception:
        return "unknown", 0.0

    try:
        return predict_regime(df)
    except Exception:
        return "unknown", 0.0


def _probabilities(label: str, confidence: float | None = None) -> Dict[str, float]:
    """Return a probability mapping for all regimes."""
    probs = {r: 0.0 for r in _ALL_REGIMES}
    if confidence is None:
        confidence = 1.0 if label in probs else 0.0
    probs[label] = confidence
    return probs


def _normalize(probs: Dict[str, float]) -> Dict[str, float]:
    """Return normalized probability mapping."""
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


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

    for col in (
        "ema20",
        "ema50",
        "adx",
        "rsi",
        "atr",
        "bb_width",
        "volume_change",
        "volume_zscore",
        "normalized_range",
    ):
        if col in df:
            df[col] = df[col].fillna(df[col].mean())

    volume_ma20 = (
        df["volume"].rolling(cfg["ma_window"]).mean()
        if len(df) >= cfg["ma_window"]
        else pd.Series(np.nan, index=df.index)
    )

    volume_jump = False
    if len(df) > 1:
        vol_change = df["volume_change"].iloc[-1]
        vol_z = df["volume_zscore"].iloc[-1]
        if (not np.isnan(vol_change) and vol_change > 1.0) or (
            not np.isnan(vol_z) and vol_z > 3
        ):
            volume_jump = True

    latest = df.iloc[-1]

    logger.debug(
        "Indicators - ADX: %.2f (trending>%.2f, sideways<%.2f), BB width: %.4f "
        "(breakout<%.2f, sideways<%.2f), RSI: %.2f (mean_rev %d-%d), EMA dist: "
        "%.4f (max %.4f), Normalized range: %.4f (volatile>%.2f)",
        latest["adx"],
        cfg["adx_trending_min"],
        cfg["adx_sideways_max"],
        latest["bb_width"],
        cfg["bb_width_breakout_max"],
        cfg["bb_width_sideways_max"],
        latest["rsi"],
        cfg["rsi_mean_rev_min"],
        cfg["rsi_mean_rev_max"],
        abs(latest["close"] - latest["ema20"]) / latest["close"],
        cfg["ema_distance_mean_rev_max"],
        latest["normalized_range"],
        cfg["normalized_range_volatility_min"],
    )

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


def _classify_all(
    df: Optional[pd.DataFrame],
    higher_df: Optional[pd.DataFrame],
    cfg: dict,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[str, Dict[str, float], Dict[str, float]] | Dict[str, str] | Tuple[str, str]:
    """Return regime label, probability mapping and patterns or labels for ``df_map``."""

    ml_min_bars = cfg.get("ml_min_bars", 20)

    if df_map is not None:
        labels: Dict[str, str] = {}
        for tf, frame in df_map.items():
            h_df = None
            if tf != cfg.get("higher_timeframe"):
                h_df = df_map.get(cfg.get("higher_timeframe"))
            label, _, _ = _classify_all(frame, h_df, cfg)
            labels[tf] = label
        if len(df_map) == 2:
            return tuple(labels[tf] for tf in df_map.keys())  # type: ignore
        return labels

    if df is None:
        return "unknown", {"unknown": 0.0}, {}

    if len(df) < ml_min_bars:
        return "unknown", _probabilities("unknown"), {}

    pattern_min = float(cfg.get("pattern_min_conf", 0.0))
    patterns = detect_patterns(df, min_conf=pattern_min)

    regime = _classify_core(df, cfg, higher_df)

    # Score regimes based on indicator result and detected patterns
    scores: Dict[str, float] = {}
    for name, strength in patterns.items():
        if strength < pattern_min:
            continue
        target, weight = PATTERN_WEIGHTS.get(name, (None, 0.0))
        if target is None:
            continue
        scores[target] = scores.get(target, 0.0) + weight * float(strength)

    regime = _classify_core(df, cfg, higher_df)
    if regime != "unknown":
        scores[regime] = scores.get(regime, 0.0) + 1.0

    if scores:
        total = sum(scores.values())
        probabilities = {r: scores.get(r, 0.0) / total for r in _ALL_REGIMES}
        regime = max(scores, key=scores.get)
        log_patterns(regime, patterns)
        return regime, probabilities, patterns

    rule_probs = _probabilities(regime)

    ml_label = "unknown"
    ml_probs = {r: 0.0 for r in _ALL_REGIMES}
    use_ml = cfg.get("use_ml_regime_classifier", False)
    if use_ml and len(df) >= ml_min_bars:
        ml_label, conf = _ml_fallback(df)
        ml_probs = _probabilities(ml_label, conf)
        if regime == "unknown" and ml_label != "unknown":
            log_patterns(ml_label, patterns)
            return ml_label, ml_probs, patterns

    if regime == "unknown":
        if cfg.get("use_ml_regime_classifier", False) and len(df) >= ml_min_bars:
            label, conf = _ml_fallback(df)
            log_patterns(label, patterns)
            return label, _normalize(_probabilities(label, conf)), patterns
        if len(df) >= ml_min_bars:
            logger.info("Skipping ML fallback \u2014 ML disabled")
        else:
            logger.info(
                "Skipping ML fallback \u2014 insufficient data (%d rows)", len(df)
            )
        return regime, _probabilities(regime, 0.0), patterns

    if ml_label != "unknown" and use_ml and len(df) >= ml_min_bars:
        weight = cfg.get("ml_blend_weight", 0.5)
        final_probs = {
            r: (1 - weight) * rule_probs.get(r, 0.0) + weight * ml_probs.get(r, 0.0)
            for r in _ALL_REGIMES
        }
        regime = max(final_probs, key=final_probs.get)
    else:
        final_probs = rule_probs

    log_patterns(regime, patterns)
    return regime, final_probs, patterns


def classify_regime(
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
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
    symbol : Optional[str], default None
        Symbol name used for adaptive threshold calculations.

    Returns
    -------
    Tuple[str, Dict[str, float]] or Tuple[str, float]
        If ``df_map`` is ``None`` the function returns ``(label, probabilities)``
        when enough history is available, where ``probabilities`` maps each
        regime to its probability.  When the ML fallback is used due to
        insufficient history it returns ``(label, confidence)`` with
        ``confidence`` in ``[0, 1]``.
    Dict[str, str] or Tuple[str, str]
        When ``df_map`` is provided the regime for each timeframe is returned.
        If exactly two timeframes are supplied the result is a tuple preserving
        ``df_map`` insertion order; otherwise a ``{timeframe: label}`` mapping
        is produced.
    """

    cfg = CONFIG if config_path is None else _load_config(Path(config_path))
    _configure_logger(cfg)
    cfg = adaptive_thresholds(cfg, df, symbol)

    ml_min_bars = cfg.get("ml_min_bars", 20)

    if df_map is None and (df is None or len(df) < ml_min_bars):
        return "unknown", set()

    result = _classify_all(df, higher_df, cfg, df_map=df_map)

    if df_map is not None:
        return result

    label, probs, _ = result
    return label, probs


def classify_regime_with_patterns(
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """Return the regime label and detected pattern scores."""

    cfg = CONFIG if config_path is None else _load_config(Path(config_path))
    _configure_logger(cfg)
    cfg = adaptive_thresholds(cfg, df, symbol)
    label, _, patterns = _classify_all(df, higher_df, cfg)
    return label, patterns


async def classify_regime_async(
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Tuple[str, object] | Dict[str, str] | Tuple[str, str]:
    """Asynchronous wrapper around :func:`classify_regime`."""
    return await asyncio.to_thread(
        classify_regime,
        df,
        higher_df,
        df_map=df_map,
        config_path=config_path,
        symbol=symbol,
    )


async def classify_regime_with_patterns_async(
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """Async wrapper around :func:`classify_regime_with_patterns`."""
    return await asyncio.to_thread(
        classify_regime_with_patterns,
        df,
        higher_df,
        config_path=config_path,
        symbol=symbol,
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
    label, info = await classify_regime_async(
        df, higher_df, config_path=config_path, symbol=symbol
    )
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
