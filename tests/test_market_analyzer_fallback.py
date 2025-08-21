import asyncio
import numpy as np
import pandas as pd
import pytest

import crypto_bot.utils.market_analyzer as ma


def _make_df(rows: int = 60) -> pd.DataFrame:
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
async def test_analyze_symbol_falls_back_to_heuristics(monkeypatch, caplog):
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ma, "ML_AVAILABLE", True)
    monkeypatch.setattr(ma, "classify_regime_async", boom)
    monkeypatch.setattr(ma, "classify_regime_cached", boom)

    df = _make_df()
    cfg = {"timeframe": "1h"}
    caplog.set_level("ERROR")

    res = await ma.analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    assert res["regime"]
    probs = res.get("probabilities", {})
    assert isinstance(probs, dict) and probs
    assert sum(probs.values()) == pytest.approx(1.0)
    assert "classify_regime_async failed" in caplog.text.lower()


@pytest.mark.asyncio
async def test_tradeable_signal_when_classifier_fails(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    async def eval_stub(strategies, df, config, max_parallel=4):
        return [(0.5, "long", 0.0) for _ in strategies]

    monkeypatch.setattr(ma, "ML_AVAILABLE", True)
    monkeypatch.setattr(ma, "classify_regime_async", boom)
    monkeypatch.setattr(ma, "evaluate_async", eval_stub)

    df = _make_df()
    cfg = {"timeframe": "1h"}

    res = await ma.analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    assert res["direction"] != "none"
    assert res["score"] != 0
