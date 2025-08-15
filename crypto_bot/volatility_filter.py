"""Helpers for evaluating market volatility."""
from __future__ import annotations

import os

import pandas as pd
import requests

from crypto_bot.indicators.atr import calc_atr as _calc_atr_series
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "volatility.log")

DEFAULT_FUNDING_URL = "https://funding.example.com"


def fetch_funding_rate(symbol: str) -> float:
    """Return the current funding rate for ``symbol``."""
    mock = os.getenv("MOCK_FUNDING_RATE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    try:
        base_url = os.getenv("FUNDING_RATE_URL", DEFAULT_FUNDING_URL)
        url = f"{base_url}{symbol}" if "?" in base_url else f"{base_url}?pair={symbol}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "result" in data and isinstance(data["result"], dict):
                first = next(iter(data["result"].values()), {})
                return float(first.get("fr", 0.0))
            if "rates" in data and isinstance(data["rates"], list) and data["rates"]:
                last = data["rates"][-1]
                if isinstance(last, dict):
                    return float(last.get("relativeFundingRate", 0.0))
                first = data["rates"][0]
                if isinstance(first, dict):
                    return float(first.get("relativeFundingRate", 0.0))
            return float(data.get("rate", 0.0))
    except Exception as exc:
        logger.error("Failed to fetch funding rate: %s", exc)
    return 0.0


def calc_atr_cached(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate the Average True Range using cached values."""
    lookback = window
    series = _calc_atr_series(df, period=window)
    cached = cache_series(f"atr_{window}", df, series, lookback)
    return float(cached.iloc[-1])


# Backwards compatibility: external code may still import ``calc_atr``
calc_atr = calc_atr_cached


def too_flat(df: pd.DataFrame, min_atr_pct: float) -> bool:
    """Return True if ATR is below ``min_atr_pct`` of price."""
    atr = calc_atr_cached(df)
    price = df["close"].iloc[-1]
    if price == 0:
        return True
    return bool(atr / price < min_atr_pct)


def too_hot(symbol: str, max_funding_rate: float) -> bool:
    """Return True when funding rate exceeds ``max_funding_rate``."""
    rate = fetch_funding_rate(symbol)
    return bool(rate > max_funding_rate)

