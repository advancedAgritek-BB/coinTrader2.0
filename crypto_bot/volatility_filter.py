"""Volatility helpers and funding-rate checks.

This module exposes lightweight utilities used by various strategies to filter
symbols based on recent volatility and funding rates.  It deliberately keeps its
dependencies minimal so importing it does not trigger heavy modules or network
calls during test collection.

Two categories of helpers are provided:

``atr_pct`` and ``too_flat`` operate on OHLCV data frames using indicator
functions from :mod:`crypto_bot.utils.indicators`.

``fetch_funding_rate`` and ``too_hot`` query (or mock) funding rates for a
symbol which some tests use to emulate external services.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import requests

from crypto_bot.utils.indicators import calc_atr as _calc_atr

# Default API used when ``FUNDING_RATE_URL`` is not provided.  This value is
# only a placeholder; tests patch the HTTP request so no real network call is
# performed.
DEFAULT_FUNDING_URL = (
    "https://futures.kraken.com/derivatives/api/v3/"
    "historical-funding-rates?symbol="
)


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range as a percentage of ``close`` price."""

    atr = _calc_atr(df, period=period)
    return (atr / df["close"]).fillna(0.0)


def too_flat(
    df: pd.DataFrame,
    atr_period: int = 14,
    threshold: float = 0.004,
) -> bool:
    """Heuristic to detect markets with very low volatility.

    The median ATR% of the last ``atr_period`` values is compared against
    ``threshold``. When insufficient data is provided the function returns
    ``True`` as a conservative default.

    Callers should normally provide ``threshold`` explicitly—typically from
    configuration—rather than relying on the default ``0.004`` which is
    retained only for backward compatibility.

    For backwards compatibility the second positional argument may be a float
    representing ``threshold`` (the old signature). In that case ``atr_period``
    defaults to ``14``.
    """

    # Backwards compatibility for legacy ``too_flat(df, threshold)`` usage.
    if isinstance(atr_period, float) and threshold == 0.004:
        threshold = atr_period
        atr_period = 14

    if len(df) < atr_period:
        return True
    ap = atr_pct(df, period=atr_period).iloc[-atr_period:].median()
    return float(ap) < threshold


def fetch_funding_rate(symbol: str) -> float:
    """Return the current funding rate for ``symbol``.

    The function honours the ``MOCK_FUNDING_RATE`` environment variable which
    allows tests to provide deterministic values.  When a real request is made
    the JSON response is parsed with best‑effort handling for several common
    shapes used by funding‑rate APIs.
    """

    mock = os.getenv("MOCK_FUNDING_RATE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0

    base_url = os.getenv("FUNDING_RATE_URL", DEFAULT_FUNDING_URL)
    url = f"{base_url}{symbol}"

    try:  # pragma: no cover - network best effort
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return 0.0

    if isinstance(data, dict):
        # Shape: {"rates": [{"relativeFundingRate": 0.01}, ...]}
        rates = data.get("rates")
        if isinstance(rates, Iterable):
            for item in reversed(list(rates)):
                if isinstance(item, dict) and "relativeFundingRate" in item:
                    try:
                        return float(item["relativeFundingRate"])
                    except (TypeError, ValueError):
                        break

        # Shape: {"result": {"symbol": {"fr": 0.01}}}
        result = data.get("result")
        if isinstance(result, dict):
            first = next(iter(result.values()), {})
            if isinstance(first, dict) and "fr" in first:
                try:
                    return float(first["fr"])
                except (TypeError, ValueError):
                    pass

        # Shape: {"rate": 0.01}
        if "rate" in data:
            try:
                return float(data["rate"])
            except (TypeError, ValueError):
                pass

    return 0.0


def too_hot(symbol: str, max_funding_rate: float) -> bool:
    """Return ``True`` when the funding rate exceeds ``max_funding_rate``."""

    return float(fetch_funding_rate(symbol)) > max_funding_rate


# Keep legacy import path working for existing callers
def calc_atr(
    df: pd.DataFrame,
    window: int = 14,
    *,
    period: int | None = None,
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> pd.Series:
    """Compatibility wrapper returning an ATR series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input OHLC data.
    window : int, default 14
        Lookback window for the ATR calculation.
    period : int, optional
        Alias for ``window`` kept for backwards compatibility.  When provided it
        takes precedence over ``window``.
    high, low, close : str
        Column names for the respective OHLC values.

    Exposes :func:`calc_atr` under the historical import while mirroring the
    original return type.
    """

    if period is not None:
        window = int(period)

    kwargs: dict[str, str] = {}
    if high != "high":
        kwargs["high"] = high
    if low != "low":
        kwargs["low"] = low
    if close != "close":
        kwargs["close"] = close

    return _calc_atr(df, window, **kwargs)


__all__ = ["atr_pct", "too_flat", "fetch_funding_rate", "too_hot", "calc_atr"]

