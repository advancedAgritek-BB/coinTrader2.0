"""Helpers for evaluating market volatility."""

from __future__ import annotations

import os

import pandas as pd
import requests

from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/volatility.log")

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
            return float(data.get("rate", 0.0))
    except Exception as exc:
        logger.error("Failed to fetch funding rate: %s", exc)
    return 0.0


def calc_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate the Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr)


def too_flat(df: pd.DataFrame, min_atr_pct: float) -> bool:
    """Return True if ATR is below ``min_atr_pct`` of price."""
    atr = calc_atr(df)
    price = df["close"].iloc[-1]
    if price == 0:
        return True
    return bool(atr / price < min_atr_pct)


def too_hot(symbol: str, max_funding_rate: float) -> bool:
    """Return True when funding rate exceeds ``max_funding_rate``."""
    rate = fetch_funding_rate(symbol)
    return bool(rate > max_funding_rate)

