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
"""Cross-chain arbitrage strategy placeholder."""

from typing import Optional, Tuple
import pandas as pd


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return (score, direction) for potential arbitrage opportunity.

    Currently a minimal implementation that always returns no signal. Integrate
    with exchange price feeds to produce real arbitrage signals.
    """
    return 0.0, "none"
from __future__ import annotations

import asyncio
from typing import Optional, Tuple, Mapping

import pandas as pd

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.solana import fetch_solana_prices
from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
) -> Tuple[float, str]:
    """Return (score, direction) based on cross-chain price spread."""
    config: Optional[Mapping[str, object]] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Mapping[str, object]] = None,
) -> Tuple[float, str]:
    """Return arbitrage signal comparing CEX OHLCV data to Solana prices."""

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
