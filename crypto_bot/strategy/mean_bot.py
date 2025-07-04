from typing import Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Score mean reversion opportunities using multiple indicators."""

    if len(df) < 3:
        return 0.0, "none"

    df = df.copy()

    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    # Bollinger Band z-score
    mean = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_z"] = (df["close"] - mean) / std

    kc = ta.volatility.KeltnerChannel(
        df["high"], df["low"], df["close"], window=20
    )
    df["kc_h"] = kc.keltner_channel_hband()
    df["kc_l"] = kc.keltner_channel_lband()

    vwap = ta.volume.VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"], window=14
    )
    df["vwap"] = vwap.volume_weighted_average_price()

    latest = df.iloc[-1]

    long_scores = []
    short_scores = []

    if latest["rsi"] < 50:
        long_scores.append(min((50 - latest["rsi"]) / 20, 1))
    elif latest["rsi"] > 50:
        short_scores.append(min((latest["rsi"] - 50) / 20, 1))

    if not pd.isna(latest["bb_z"]):
        if latest["bb_z"] < 0:
            long_scores.append(min(-latest["bb_z"] / 2, 1))
        elif latest["bb_z"] > 0:
            short_scores.append(min(latest["bb_z"] / 2, 1))

    if not pd.isna(latest["kc_h"]) and not pd.isna(latest["kc_l"]):
        width = latest["kc_h"] - latest["kc_l"]
        if width > 0:
            if latest["close"] < latest["kc_l"]:
                long_scores.append(min((latest["kc_l"] - latest["close"]) / width, 1))
            elif latest["close"] > latest["kc_h"]:
                short_scores.append(min((latest["close"] - latest["kc_h"]) / width, 1))

    if not pd.isna(latest["vwap"]):
        diff = abs(latest["close"] - latest["vwap"]) / latest["vwap"]
        if latest["close"] < latest["vwap"]:
            long_scores.append(min(diff / 0.05, 1))
        elif latest["close"] > latest["vwap"]:
            short_scores.append(min(diff / 0.05, 1))

    long_score = sum(long_scores) / len(long_scores) if long_scores else 0.0
    short_score = sum(short_scores) / len(short_scores) if short_scores else 0.0

    if long_score > short_score and long_score > 0:
        score = long_score
        direction = "long"
    elif short_score > long_score and short_score > 0:
        score = short_score
        direction = "short"
    else:
        return 0.0, "none"

    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    return float(max(0.0, min(score, 1.0))), direction
