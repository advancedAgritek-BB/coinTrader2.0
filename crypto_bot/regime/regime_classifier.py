from pathlib import Path
from typing import Optional, Tuple, Dict
import asyncio
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

    df["ema20"] = (
        ta.trend.ema_indicator(df["close"], window=cfg["ema_fast"])
        if len(df) >= cfg["ema_fast"]
        else np.nan
    )
    df["ema50"] = (
        ta.trend.ema_indicator(df["close"], window=cfg["ema_slow"])
        if len(df) >= cfg["ema_slow"]
        else np.nan
    )

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
            df["adx"] = np.nan
            df["rsi"] = np.nan
            df["atr"] = np.nan
            return "unknown"
    else:
        df["adx"] = np.nan
        df["rsi"] = np.nan
        df["atr"] = np.nan

    if len(df) >= cfg["bb_window"]:
        bb = ta.volatility.BollingerBands(df["close"], window=cfg["bb_window"])
        df["bb_width"] = bb.bollinger_wband()
    else:
        df["bb_width"] = np.nan

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


