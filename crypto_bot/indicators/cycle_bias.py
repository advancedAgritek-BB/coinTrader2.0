"""Market cycle bias derived from on-chain metrics."""

from __future__ import annotations

import os
import requests

from crypto_bot.utils.logger import indicator_logger

DEFAULT_MVRV_URL = "https://api.example.com/mvrv"
DEFAULT_NUPL_URL = "https://api.example.com/nupl"
DEFAULT_SOPR_URL = "https://api.example.com/sopr"


def _fetch_value(url: str, mock_env: str) -> float:
    """Return metric value from ``url`` using ``mock_env`` for tests."""
    mock = os.getenv(mock_env)
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return float(data.get("value", 0.0))
    except Exception as exc:
        indicator_logger.error("Failed to fetch metric from %s: %s", url, exc)
    return 0.0


def get_cycle_bias(config: dict | None = None) -> float:
    """Return bias multiplier (>1 bullish, <1 bearish)."""
    cfg = config or {}
    mvrv_url = cfg.get("mvrv_url") or os.getenv("MVRV_URL", DEFAULT_MVRV_URL)
    nupl_url = cfg.get("nupl_url") or os.getenv("NUPL_URL", DEFAULT_NUPL_URL)
    sopr_url = cfg.get("sopr_url") or os.getenv("SOPR_URL", DEFAULT_SOPR_URL)

    mvrv = _fetch_value(mvrv_url, "MOCK_MVRV")
    nupl = _fetch_value(nupl_url, "MOCK_NUPL")
    sopr = _fetch_value(sopr_url, "MOCK_SOPR")

    score = (mvrv + nupl + sopr) / 3
    bias = 1 + score
    indicator_logger.info(
        "Cycle bias %.2f from MVRV %.2f NUPL %.2f SOPR %.2f",
        bias,
        mvrv,
        nupl,
        sopr,
    )
    return bias
