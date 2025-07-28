from __future__ import annotations

import asyncio
from typing import Optional, Tuple, Mapping

import pandas as pd

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.solana import fetch_solana_prices
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Mapping[str, object]] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Mapping[str, object]] = None,
) -> Tuple[float, str]:
    """Return arbitrage signal comparing CEX OHLCV data to Solana prices."""

    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("cross_chain_arb_bot", {}) if config else {}
    pair = str(params.get("pair", "SOL/USDC"))
    threshold = float(params.get("spread_threshold", 0.0))

    cfg = mempool_cfg or {}
    if mempool_monitor and cfg.get("enabled"):
        try:
            fee_thr = float(cfg.get("suspicious_fee_threshold", 0.0))
            if mempool_monitor.is_suspicious(fee_thr):
                return 0.0, "none"
        except Exception:
            return 0.0, "none"

    try:
        prices = asyncio.run(fetch_solana_prices([pair]))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        prices = loop.run_until_complete(fetch_solana_prices([pair]))
    dex_price = prices.get(pair)
    if dex_price is None or dex_price <= 0:
        return 0.0, "none"

    cex_price = float(df["close"].iloc[-1])
    if cex_price <= 0:
        return 0.0, "none"

    diff = (dex_price - cex_price) / cex_price
    if abs(diff) < threshold:
        return 0.0, "none"

    score = min(abs(diff), 1.0)
    score = normalize_score_by_volatility(df, score)
    direction = "long" if diff > 0 else "short"
    return score, direction
