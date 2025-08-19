import asyncio
import numpy as np
import pandas as pd
import pytest

import crypto_bot.utils.market_analyzer as ma
import crypto_bot.sentiment_filter as sf


def make_df(volume: float, rows: int = 60) -> pd.DataFrame:
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    vol = np.full(rows, volume)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol})


def _setup(monkeypatch):
    async def fake_async(*_a, **_k):
        return "trending", {"trending": 1.0}

    async def fake_cached(*_a, **_k):
        return "trending", 1.0

    async def fake_patterns(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_async", fake_async)
    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "classify_regime_with_patterns_async", fake_patterns)
    monkeypatch.setattr(ma, "detect_patterns", lambda _df: {})
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: pd.Series([0.0]))
    monkeypatch.setattr(ma, "strategy_name", lambda *a, **k: "dummy")
    monkeypatch.setattr(ma, "route", lambda *a, **k: (lambda df, cfg=None: (0.5, "long")))

    async def fake_eval(*_a, **_k):
        return [(0.5, "long", None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)

    async def fake_boost(*_a, **_k):
        return 1.0

    monkeypatch.setattr(sf, "boost_factor", fake_boost)


@pytest.mark.asyncio
async def test_volume_normalization(monkeypatch):
    _setup(monkeypatch)
    cfg = {"timeframe": "1h"}
    low_df = make_df(10)
    high_df = make_df(10000)
    res_low = await ma.analyze_symbol("AAA", {"1h": low_df}, "cex", cfg, None)
    res_high = await ma.analyze_symbol("AAA", {"1h": high_df}, "cex", cfg, None)
    expected_low = 0.5 / (1 + np.log1p(10) / 10)
    expected_high = 0.5 / (1 + np.log1p(10000) / 10)
    assert res_low["score"] == pytest.approx(expected_low)
    assert res_high["score"] == pytest.approx(expected_high)
    assert res_low["score"] > res_high["score"]
