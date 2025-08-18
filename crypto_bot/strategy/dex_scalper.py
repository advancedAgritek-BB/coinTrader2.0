import os
import asyncio
import pandas as pd
from typing import Tuple, Optional
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.pair_cache import load_liquid_pairs
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor

ALLOWED_PAIRS = load_liquid_pairs() or []


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Tuple[float, str]:
    """Short-term momentum strategy using EMA divergence on DEX pairs.

    When ``mempool_monitor`` is supplied the current Solana priority fee is
    used. If not provided, the ``MOCK_PRIORITY_FEE`` environment variable can
    define the fee for testing.
    """
    if df.empty:
        return 0.0, "none"

    params = config.get("dex_scalper", {}) if config else {}
    fast_window = params.get("ema_fast", 3)
    slow_window = params.get("ema_slow", 10)
    min_score = params.get("min_signal_score", 0.1)
    fee_cap = params.get("priority_fee_cap_micro_lamports", 10.0)

    if fee_cap > 0:
        fee: float | None = None
        if mempool_monitor is not None:
            try:
                fee = asyncio.run(mempool_monitor.fetch_priority_fee())
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    fee = loop.run_until_complete(
                        mempool_monitor.fetch_priority_fee()
                    )
                except Exception:
                    fee = None
            except Exception:
                fee = None
        if fee is None:
            mock_fee = os.getenv("MOCK_PRIORITY_FEE")
            if mock_fee is not None:
                try:
                    fee = float(mock_fee)
                except ValueError:
                    fee = None
        if fee is not None and fee > fee_cap:
            return 0.0, "none"

    if len(df) < slow_window:
        return 0.0, "none"

    lookback = slow_window

    ema_fast_full = ta.trend.ema_indicator(df["close"], window=fast_window)
    ema_slow_full = ta.trend.ema_indicator(df["close"], window=slow_window)
    ema_fast = cache_series("ema_fast", df, ema_fast_full, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow_full, lookback)

    recent = df.iloc[-(lookback + 1) :].copy()
    recent["ema_fast"] = ema_fast
    recent["ema_slow"] = ema_slow
    df = recent

    latest = df.iloc[-1]
    if (
        latest["close"] == 0
        or pd.isnull(latest["ema_fast"])
        or pd.isnull(latest["ema_slow"])
    ):
        return 0.0, "none"

    momentum = latest["ema_fast"] - latest["ema_slow"]
    score = min(abs(momentum) / latest["close"], 1.0)

    if score < min_score:
        return 0.0, "none"

    if momentum > 0:
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    elif momentum < 0:
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"
    return 0.0, "none"


class regime_filter:
    """DEX scalper works for any regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return True
