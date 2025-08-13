from pathlib import Path
from typing import Dict, Optional, Tuple
import asyncio
import time
import logging
import os

import pandas as pd
import numpy as np
import ta
import yaml

from crypto_bot.utils.telegram import TelegramNotifier

from .pattern_detector import detect_patterns
from crypto_bot.utils.pattern_logger import log_patterns
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils import timeframe_seconds
from crypto_bot.utils.telemetry import telemetry
from crypto_bot.utils.telegram import TelegramNotifier


# Thresholds and ML blend settings are defined in ``regime_config.yaml``
CONFIG_PATH = Path(__file__).with_name("regime_config.yaml")


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = _load_config(CONFIG_PATH)


logger = setup_logger(__name__, LOG_DIR / "bot.log")


def _log_prediction(action: str, score: float, meta: dict | None = None) -> None:
    """Log prediction details and track fallback usage."""
    meta = meta or {}
    source = meta.get("source")
    regime = meta.get("regime")
    if source == "fallback":
        telemetry.inc("ml_fallbacks")
    extra = f" regime={regime}" if regime else ""
    logger.info(
        "ML predict action=%s score=%.4f source=%s%s",
        action,
        score,
        source,
        extra,
    )


def _configure_logger(cfg: dict) -> None:
    level_str = str(cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)


_configure_logger(CONFIG)

_supabase_model = None
_supabase_model_lock = asyncio.Lock()
_model_lock = asyncio.Lock()
_ml_recovery_task: asyncio.Task | None = None

_ALL_REGIMES = [
    "trending",
    "sideways",
    "mean-reverting",
    "dip_hunter",
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


def is_ml_available() -> bool:
    """Return ``True`` if ML dependencies and credentials are available."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        return False
    try:  # pragma: no cover - optional dependency
        import lightgbm  # noqa: F401
        from supabase import create_client  # noqa: F401
    except Exception:
        return False
    return True


def _apply_hft_overrides(cfg: dict, timeframe: Optional[str]) -> dict:
    """Return a copy of ``cfg`` with HFT overrides applied for ``timeframe``."""

    if not timeframe or timeframe.lower() != "30s":
        return cfg

    out = cfg.copy()
    for key, value in cfg.items():
        if key.startswith("hft_"):
            out[key[4:]] = value
    return out


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
                df["high"],
                df["low"],
                df["close"],
                window=cfg.get("indicator_window", 14),
            )
            avg_atr = float(atr.mean())
            factor = min(2.0, avg_atr / float(baseline)) if baseline else 1.0
            out["adx_trending_min"] = cfg["adx_trending_min"] * factor
            out["normalized_range_volatility_min"] = (
                cfg["normalized_range_volatility_min"] * factor
            )
        except Exception:
            pass

    try:  # pragma: no cover - optional dependency
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.ar_model import AutoReg
    except ImportError as exc:
        logger.warning("statsmodels unavailable; drift detection disabled: %s", exc)
    else:
        try:
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


def _ml_fallback(
    df: pd.DataFrame, notifier: TelegramNotifier | None = None
) -> Tuple[str, float]:
    """Return regime label and confidence using a Supabase model with fallback."""
    try:  # pragma: no cover - Supabase model is optional
        from .ml_regime_model import predict_regime as sb_predict

        result = sb_predict(df)
        if isinstance(result, tuple):
            label, conf = result
        else:
            label, conf = result, 1.0

        _log_prediction(label, float(conf), {"source": "registry", "regime": label})
        return label, float(conf)
    except Exception as exc:  # pragma: no cover - log and fallback
        logger.error("%s", exc)
        logger.warning("ML model unavailable; using fallback")
        if notifier is not None:
            try:
                notifier.notify("⚠️ ML model unavailable; using fallback")
            except Exception:
                pass
        logger.info("Falling back to embedded model")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            global _ml_recovery_task
            if _ml_recovery_task is None or _ml_recovery_task.done():
                _ml_recovery_task = loop.create_task(
                    _ml_recovery_loop(notifier)
                )

    try:  # pragma: no cover - optional dependency
        from .ml_fallback import predict_regime
    except Exception:
        return "unknown", 0.0

    try:
        label, conf = predict_regime(df)
    except Exception:
        return "unknown", 0.0
    _log_prediction(label, float(conf), {"source": "fallback", "regime": label})
    return label, float(conf)


async def _download_supabase_model():
    """Download LightGBM model from Supabase and return a Booster."""
    async with _supabase_model_lock:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            logger.error("Missing Supabase credentials")
            return None
        try:  # pragma: no cover - optional dependency
            from supabase import create_client
        except Exception as exc:  # pragma: no cover - log import failure
            logger.error("Supabase client unavailable: %s", exc)
            return None

        try:
            client = await asyncio.to_thread(create_client, url, key)
            file_name = os.getenv("SUPABASE_MODEL_FILE", "regime_lgbm.pkl")
            bucket = client.storage.from_("models")
            data = await asyncio.to_thread(bucket.download, file_name)
            path = Path(__file__).with_name(file_name)
            await asyncio.to_thread(path.write_bytes, data)
            import lightgbm as lgb  # pragma: no cover - optional dependency

            model = await asyncio.to_thread(lgb.Booster, model_file=str(path))
            logger.info("Downloaded %s from Supabase", file_name)
            return model
        except Exception as exc:
            logger.error("Failed to download Supabase model: %s", exc)
            return None


async def _get_supabase_model() -> object | None:
    """Return the cached Supabase model, downloading it if needed."""
    global _supabase_model
    async with _model_lock:
        if _supabase_model is None:
            _supabase_model = await asyncio.to_thread(_download_supabase_model)
        return _supabase_model


async def _ml_recovery_loop(notifier: TelegramNotifier | None) -> None:
    """Periodically attempt to restore the Supabase ML model."""
    global _supabase_model, _ml_recovery_task
    while True:
        await asyncio.sleep(3600)
        if not is_ml_available():
            continue
        model = await _download_supabase_model()
        if model is None:
            continue
        _supabase_model = model
        logger.info("Supabase ML model reloaded")
        if notifier is not None:
            try:
                await notifier.notify_async("Supabase ML model reloaded")
            except Exception:  # pragma: no cover - notifier errors aren't critical
                logger.exception("Failed to send Telegram notification")
        _ml_recovery_task = None
        return


def _classify_ml(
    df: pd.DataFrame, notifier: TelegramNotifier | None = None
) -> Tuple[str, float]:
    """Predict regime using the Supabase model with fallback."""
    try:  # pragma: no cover - optional dependency
        import lightgbm as lgb
    except ImportError as exc:
        logger.warning(
            "lightgbm unavailable; ML-based classification disabled: %s", exc
        )
        try:
            return _ml_fallback(df, notifier)
        except TypeError:
            return _ml_fallback(df)
        return _ml_fallback(df, notifier)

    if _supabase_model is None:
        # Running the async download in a blocking manner; callers should
        # prefer :func:`classify_regime_async` to avoid blocking the event loop.
        _supabase_model = asyncio.run(_download_supabase_model())

    model = _supabase_model
    model = asyncio.run(_get_supabase_model())
    if model is None:
        try:
            return _ml_fallback(df, notifier)
        except TypeError:
            return _ml_fallback(df)
        return _ml_fallback(df, notifier)

    try:
        change = df["close"].iloc[-1] - df["close"].iloc[0]
        X = np.array([[change]], dtype=float)
        prob = float(model.predict(X)[0])
        if prob > 0.55:
            label = "trending"
        elif prob < 0.45:
            label = "mean-reverting"
        else:
            label = "sideways"
        conf = abs(prob - 0.5) * 2
        _log_prediction(label, conf, {"source": "registry", "regime": label})
        return label, conf
    except Exception:
        try:
            return _ml_fallback(df, notifier)
        except TypeError:
            return _ml_fallback(df)
        return _ml_fallback(df, notifier)


def _probabilities(label: str, confidence: float | None = None) -> Dict[str, float]:
    """Return a probability mapping for all regimes."""
    probs = {r: 0.0 for r in _ALL_REGIMES}
    if confidence is None:
        confidence = 1.0 if label in probs else 0.0
    probs[label] = confidence
    return probs


def _normalize(probs: Dict[str, float], eps: float = 1e-8) -> Dict[str, float]:
    """Return normalized probability mapping."""
    total = sum(probs.values())
    if total <= 0:
        total = eps
    return {k: v / total for k, v in probs.items()}


# Cache for indicator computations keyed by ``(symbol, timeframe)``
# storing ``(last_timestamp, dataframe_with_indicators)``. This avoids
# recomputing expensive TA indicators when the input data has not changed.
_indicator_cache: Dict[Tuple[str, str], Tuple[int, pd.DataFrame]] = {}


def _compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return ``df`` with required technical indicators computed."""
    df = df.copy()
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
            return "trending"
        df["adx"] = ta.trend.adx(
            df["high"], df["low"], df["close"], window=cfg["indicator_window"]
        )
        df["rsi"] = ta.momentum.rsi(df["close"], window=cfg["indicator_window"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=cfg["indicator_window"]
        )
        df["normalized_range"] = (df["high"] - df["low"]) / df["atr"]
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

    return df


def _classify_core(
    data: pd.DataFrame,
    cfg: dict,
    higher_df: Optional[pd.DataFrame] = None,
    cache_key: Optional[Tuple[str, str]] = None,
) -> str:
    if data is None or data.empty or len(data) < 20:
        return "trending"

    ts = int(data["timestamp"].iloc[-1]) if "timestamp" in data.columns else len(data)

    if cache_key is not None:
        cached = _indicator_cache.get(cache_key)
        if cached and cached[0] == ts:
            df = cached[1].copy()
        else:
            try:
                df = _compute_indicators(data, cfg)
            except IndexError:
                return "unknown"
            _indicator_cache[cache_key] = (ts, df.copy())
    else:
        try:
            df = _compute_indicators(data, cfg)
        except IndexError:
            return "unknown"

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

    trending = (
        latest["adx"] > cfg.get("adx_trending_min", 20)
        and latest["ema20"] > latest["ema50"]
    )

    if trending and cfg.get("confirm_trend_with_higher_tf", False):
        if higher_df is None:
            trending = False
        else:
            confirm_cfg = cfg.copy()
            confirm_cfg["confirm_trend_with_higher_tf"] = False
            confirm_key = None
            if cache_key is not None and cfg.get("higher_timeframe"):
                confirm_key = (cache_key[0], str(cfg.get("higher_timeframe")))
            if (
                _classify_core(higher_df, confirm_cfg, None, cache_key=confirm_key)
                != "trending"
            ):
                trending = False

    regime = "unknown"

    squeeze = (
        latest["bb_width"] < 0.05
        and not np.isnan(volume_ma20.iloc[-1])
        and latest["volume"] > volume_ma20.iloc[-1] * cfg["breakout_volume_mult"]
    )
    if not squeeze:
        logger.debug("No squeeze")

    if squeeze or volume_jump:
        regime = "breakout"
    elif trending:
        regime = "trending"
    elif not trending and latest["adx"] < cfg.get("dip_hunter_adx_max", 25):
        regime = "dip_hunter"
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
    cache_key: Optional[Tuple[str, str]] = None,
    notifier: TelegramNotifier | None = None,
) -> Tuple[str, Dict[str, float], Dict[str, float]] | Dict[str, str] | Tuple[str, str]:
    """Return regime label, probability mapping and patterns or labels for ``df_map``."""

    ml_min_bars = cfg.get("ml_min_bars", 10)

    if df_map is not None:
        labels: Dict[str, str] = {}
        for tf, frame in df_map.items():
            h_df = None
            if tf != cfg.get("higher_timeframe"):
                h_df = df_map.get(cfg.get("higher_timeframe"))
            label, _, _ = _classify_all(
                frame, h_df, cfg, cache_key=None, notifier=notifier
            )
            labels[tf] = label
        if len(df_map) == 2:
            return tuple(labels[tf] for tf in df_map.keys())  # type: ignore
        return labels

    if df is None:
        return "unknown", {"unknown": 0.0}, {}

    if len(df) < ml_min_bars:
        pattern_min = float(cfg.get("pattern_min_conf", 0.0))
        patterns = detect_patterns(df, min_conf=pattern_min)
        label = "breakout" if patterns.get("breakout", 0.0) > 0 else "trending"
        log_patterns(label, patterns)
        return label, _probabilities(label), patterns

    pattern_min = float(cfg.get("pattern_min_conf", 0.0))
    patterns = detect_patterns(df, min_conf=pattern_min)

    regime = _classify_core(df, cfg, higher_df, cache_key=cache_key)
    if regime == "unknown":
        use_ml = cfg.get("use_ml_regime_classifier", False)
        if use_ml and len(df) >= ml_min_bars:
            label, conf = _classify_ml(df, notifier)
            log_patterns(label, patterns)
            return label, _probabilities(label, conf), patterns
        if len(df) >= ml_min_bars:
            logger.info("Skipping ML fallback \u2014 ML disabled")
        else:
            logger.info(
                "Skipping ML fallback \u2014 insufficient data (%d rows)", len(df)
            )
        return regime, _probabilities(regime, 0.0), patterns

    # Score regimes based on indicator result and detected patterns
    scores: Dict[str, float] = {}
    for name, strength in patterns.items():
        if strength < pattern_min:
            continue
        target, weight = PATTERN_WEIGHTS.get(name, (None, 0.0))
        if target is None:
            continue
        scores[target] = scores.get(target, 0.0) + weight * float(strength)

    scores[regime] = scores.get(regime, 0.0) + 1.0

    total = sum(scores.values())
    probabilities = {r: scores.get(r, 0.0) / total for r in _ALL_REGIMES}
    regime = max(scores, key=scores.get)

    rule_probs = _probabilities(regime)

    ml_label = "unknown"
    ml_probs = {r: 0.0 for r in _ALL_REGIMES}
    use_ml = cfg.get("use_ml_regime_classifier", False)
    if use_ml and len(df) >= ml_min_bars:
        ml_label, conf = _classify_ml(df, notifier)
        ml_probs = _probabilities(ml_label, conf)
        if regime == "unknown" and ml_label != "unknown":
            log_patterns(ml_label, patterns)
            return ml_label, ml_probs, patterns

    if regime == "unknown":
        if cfg.get("use_ml_regime_classifier", False) and len(df) >= ml_min_bars:
            label, conf = _classify_ml(df, notifier)
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
    timeframe: Optional[str] = None,
    cache_key: Optional[Tuple[str, str]] = None,
    notifier: TelegramNotifier | None = None,
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
    timeframe : Optional[str], default None
        Timeframe string used to adjust thresholds for sub-minute data.

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

    cfg = CONFIG.copy()
    if config_path is not None:
        cfg.update(_load_config(Path(config_path)))
    _configure_logger(cfg)
    cfg = _apply_hft_overrides(cfg, timeframe)
    cfg = adaptive_thresholds(cfg, df, symbol)

    if timeframe and timeframe_seconds(None, timeframe) < 60:
        cfg = cfg.copy()
        cfg["adx_trending_min"] = float(
            os.getenv("HFT_ADX_MIN")
            or cfg.get("hft_adx_trending_min", cfg["adx_trending_min"])
        )
        cfg["rsi_mean_rev_min"] = float(
            os.getenv("HFT_RSI_MIN")
            or cfg.get("hft_rsi_mean_rev_min", cfg["rsi_mean_rev_min"])
        )
        cfg["rsi_mean_rev_max"] = float(
            os.getenv("HFT_RSI_MAX")
            or cfg.get("hft_rsi_mean_rev_max", cfg["rsi_mean_rev_max"])
        )
        cfg["normalized_range_volatility_min"] = float(
            os.getenv("HFT_NR_VOL_MIN")
            or cfg.get(
                "hft_normalized_range_volatility_min",
                cfg["normalized_range_volatility_min"],
            )
        )
        cfg["indicator_window"] = int(
            os.getenv("HFT_INDICATOR_WINDOW")
            or cfg.get("hft_indicator_window", cfg["indicator_window"])
        )
        cfg["ml_blend_weight"] = float(
            os.getenv("HFT_ML_BLEND_WEIGHT")
            or cfg.get("hft_ml_blend_weight", cfg.get("ml_blend_weight", 0.7))
        )

    ml_min_bars = cfg.get("ml_min_bars", 10)

    if df_map is None and df is None:
        return "unknown", {"unknown": 0.0}

    kwargs = dict(df_map=df_map)
    if cache_key is not None:
        kwargs["cache_key"] = cache_key
    if notifier is not None:
        kwargs["notifier"] = notifier
    result = _classify_all(df, higher_df, cfg, **kwargs)
    result = _classify_all(
        df, higher_df, cfg, df_map=df_map, cache_key=cache_key, notifier=notifier
    )

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
    timeframe: Optional[str] = None,
    notifier: TelegramNotifier | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Return the regime label and detected pattern scores."""

    cfg = CONFIG.copy()
    if config_path is not None:
        cfg.update(_load_config(Path(config_path)))
    _configure_logger(cfg)
    cfg = _apply_hft_overrides(cfg, timeframe)
    cfg = adaptive_thresholds(cfg, df, symbol)
    kwargs = {}
    if notifier is not None:
        kwargs["notifier"] = notifier
    label, _, patterns = _classify_all(df, higher_df, cfg, **kwargs)
    label, _, patterns = _classify_all(df, higher_df, cfg, notifier=notifier)
    return label, patterns


async def classify_regime_async(
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    cache_key: Optional[Tuple[str, str]] = None,
    notifier: TelegramNotifier | None = None,
) -> Tuple[str, object] | Dict[str, str] | Tuple[str, str]:
    """Asynchronous wrapper around :func:`classify_regime`."""
    return await asyncio.to_thread(
        classify_regime,
        df,
        higher_df,
        df_map=df_map,
        config_path=config_path,
        symbol=symbol,
        timeframe=timeframe,
        cache_key=cache_key,
        notifier=notifier,
    )


async def classify_regime_with_patterns_async(
    df: pd.DataFrame,
    higher_df: Optional[pd.DataFrame] = None,
    *,
    config_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    notifier: TelegramNotifier | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Async wrapper around :func:`classify_regime_with_patterns`."""
    return await asyncio.to_thread(
        classify_regime_with_patterns,
        df,
        higher_df,
        config_path=config_path,
        symbol=symbol,
        timeframe=timeframe,
        notifier=notifier,
    )


# Caching utilities -----------------------------------------------------

regime_cache: Dict[tuple[str, str], str] = {}
_regime_cache_ts: Dict[tuple[str, str], int] = {}
_regime_cache_lock = asyncio.Lock()


async def classify_regime_cached(
    symbol: str,
    timeframe: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    higher_df: Optional[pd.DataFrame] = None,
    profile: bool = False,
    *,
    config_path: Optional[str] = None,
    notifier: TelegramNotifier | None = None,
) -> Tuple[str, object]:
    """Classify ``symbol`` regime with caching and optional profiling."""

    if df is None or df.empty:
        return "unknown", 0.0

    ts = int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else len(df)
    key = (symbol, timeframe or "")
    async with _regime_cache_lock:
        if key in regime_cache and _regime_cache_ts.get(key) == ts:
            label = regime_cache[key]
            # Info is not cached; recompute minimal patterns for compatibility
            return label, set()

    start = time.perf_counter() if profile else 0.0
    label, info = await classify_regime_async(
        df,
        higher_df,
        config_path=config_path,
        symbol=symbol,
        timeframe=timeframe,
        cache_key=key,
        notifier=notifier,
    )
    async with _regime_cache_lock:
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


async def _clear_regime_cache(symbol: str, timeframe: str) -> None:
    async with _regime_cache_lock:
        regime_cache.pop((symbol, timeframe), None)
        _regime_cache_ts.pop((symbol, timeframe), None)


def clear_regime_cache(symbol: str, timeframe: str) -> None:
    """Remove cached regime entry for ``symbol`` and ``timeframe``."""
    regime_cache.pop((symbol, timeframe), None)
    _regime_cache_ts.pop((symbol, timeframe), None)


def clear_indicator_cache(symbol: str, timeframe: str) -> None:
    """Remove cached indicator entry for ``symbol`` and ``timeframe``."""
    _indicator_cache.pop((symbol, timeframe), None)
    asyncio.run(_clear_regime_cache(symbol, timeframe))
