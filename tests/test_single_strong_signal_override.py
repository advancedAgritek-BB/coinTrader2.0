import asyncio
import numpy as np
import pandas as pd
import pytest

import crypto_bot.utils.market_analyzer as ma
from crypto_bot import strategy_router


def _make_trending_df(rows: int = 50) -> pd.DataFrame:
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    volume = np.arange(rows) + 100
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.mark.asyncio
async def test_single_strong_signal_override_dry_run(monkeypatch):
    df = _make_trending_df()

    def base(df, cfg=None):
        return 0.4, "short"

    def strong(df, cfg=None):
        return 0.7, "long"

    monkeypatch.setattr(ma, "route", lambda *a, **k: base)
    monkeypatch.setattr(strategy_router, "route", lambda *a, **k: base)
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"s": strong}.get(n))
    monkeypatch.setattr(strategy_router, "get_strategy_by_name", lambda n: {"s": strong}.get(n))

    cfg = {
        "timeframe": "1h",
        "regime_timeframes": ["1h"],
        "voting_strategies": {"strategies": ["s"], "min_agreeing_votes": 2},
    }
    res = await ma.analyze_symbol("AAA", {"1h": df}, "dry_run", cfg, None)
    assert res["direction"] == "long"


@pytest.mark.asyncio
async def test_single_strong_signal_live_respects_quorum(monkeypatch):
    df = _make_trending_df()

    def base(df, cfg=None):
        return 0.4, "short"

    def strong(df, cfg=None):
        return 0.7, "long"

    monkeypatch.setattr(ma, "route", lambda *a, **k: base)
    monkeypatch.setattr(strategy_router, "route", lambda *a, **k: base)
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"s": strong}.get(n))
    monkeypatch.setattr(strategy_router, "get_strategy_by_name", lambda n: {"s": strong}.get(n))

    cfg = {
        "timeframe": "1h",
        "regime_timeframes": ["1h"],
        "voting_strategies": {"strategies": ["s"], "min_agreeing_votes": 2},
    }
    res = await ma.analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)
    assert res["direction"] == "none"
