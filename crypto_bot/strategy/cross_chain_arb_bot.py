from __future__ import annotations

"""Simple cross-chain arbitrage strategy."""

import asyncio
from typing import Optional, Tuple, Mapping, List, Dict

import pandas as pd

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.solana import fetch_solana_prices
from crypto_bot.utils.volatility import normalize_score_by_volatility

NAME = "cross_chain_arb_bot"

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
    config: Optional[Mapping[str, object]] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Mapping[str, object]] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_,
) -> Tuple[float, str]:
    """Return arbitrage signal comparing CEX OHLCV data to Solana prices."""
    if df is None or df.empty:
        return 0.0, "none"

    config = config or {}
    params = config.get("cross_chain_arb_bot", {})
    pair = str(params.get("pair", ""))
    try:
        threshold = float(params.get("spread_threshold", 0.0))
    except (TypeError, ValueError):
        threshold = 0.0

    symbol = config.get("symbol", "")
    if symbol and not pair:
        pair = str(symbol)
    if not pair:
        return 0.0, "none"

    cfg = mempool_cfg or config.get("mempool_monitor", {})
    if mempool_monitor and cfg.get("enabled"):
        try:
            fee_thr = float(cfg.get("suspicious_fee_threshold", 0.0))
        except (TypeError, ValueError):
            fee_thr = 0.0
        try:
            suspicious = asyncio.run(mempool_monitor.is_suspicious(fee_thr))
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                suspicious = loop.run_until_complete(
                    mempool_monitor.is_suspicious(fee_thr)
                )
            except Exception:
                suspicious = False
        except Exception:
            suspicious = False
        if suspicious:
            return 0.0, "none"

    try:
        prices = asyncio.run(fetch_solana_prices([pair]))
    except RuntimeError:  # pragma: no cover - event loop already running
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
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if diff > 0 else "short"
    return score, direction


class regime_filter:
    """Match sideways and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:  # pragma: no cover - simple
        return regime in {"sideways", "volatile"}

