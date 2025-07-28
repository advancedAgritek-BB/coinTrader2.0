from __future__ import annotations

"""Simple cross-chain arbitrage strategy."""

import asyncio
from typing import Optional, Tuple, List, Dict

import pandas as pd

from crypto_bot.solana import fetch_solana_prices
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


def _fetch_prices(symbols: List[str]) -> Dict[str, float]:
    """Return Solana prices synchronously."""
    if not symbols:
        return {}
    try:  # pragma: no cover - best effort
        return asyncio.run(fetch_solana_prices(symbols))
    except Exception:
        return {}


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
) -> Tuple[float, str]:
    """Return (score, direction) based on cross-chain price spread."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("cross_chain_arb_bot", {}) if config else {}
    pairs: List[str] = params.get("pairs") or []
    threshold = float(params.get("spread_threshold", 0.0))

    mp_cfg = config.get("mempool_monitor", {}) if config else {}
    if mempool_monitor and mp_cfg.get("enabled"):
        thr = mp_cfg.get("suspicious_fee_threshold", 0.0)
        if mempool_monitor.is_suspicious(thr):
            return 0.0, "none"

    symbol = config.get("symbol") if config else None
    if symbol:
        pairs = [symbol]
    if not pairs:
        return 0.0, "none"

    prices = _fetch_prices(pairs)
    dex_price = prices.get(pairs[0])
    if dex_price is None or dex_price <= 0:
        return 0.0, "none"

    cex_price = df["close"].iloc[-1]
    if cex_price <= 0:
        return 0.0, "none"

    spread = (dex_price - cex_price) / cex_price
    if abs(spread) < threshold:
        return 0.0, "none"

    score = min(abs(spread) / threshold, 1.0)
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if spread > 0 else "short"
    return score, direction


class regime_filter:
    """Match sideways and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"sideways", "volatile"}



