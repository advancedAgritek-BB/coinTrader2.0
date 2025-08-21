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
async def test_unknown_regime_falls_back_to_heuristic(monkeypatch):
    async def unknown_async(*args, **kwargs):
        return "unknown", {"unknown": 1.0}

    def unknown_cached(*args, **kwargs):
        return "unknown", {"unknown": 1.0}

    async def eval_stub(strategies, df, config, max_parallel=4):
        return [(0.0, "none", 0.0) for _ in strategies]

    monkeypatch.setattr(ma, "ML_AVAILABLE", True)
    monkeypatch.setattr(ma, "classify_regime_async", unknown_async)
    monkeypatch.setattr(ma, "classify_regime_cached", unknown_cached)
    monkeypatch.setattr(ma, "evaluate_async", eval_stub)

    df = _make_df()
    cfg = {"timeframe": "1h"}

    res = await ma.analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    assert res["regime"] != "unknown"
    assert res.get("confidence", 0) > 0
