import asyncio
import numpy as np
import pandas as pd

from crypto_bot.models import price_predictor as pp
import crypto_bot.utils.market_analyzer as ma


def _df(n: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "open": rng.rand(n) + 100,
        "high": rng.rand(n) + 101,
        "low": rng.rand(n) + 99,
        "close": rng.rand(n) + 100,
    })


def test_predict_score(monkeypatch):
    df = _df()
    if pp.torch is not None and hasattr(pp.torch, "no_grad"):
        class Dummy(pp.PriceNet):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True)
        dummy = Dummy()
        score = pp.predict_score(df, model=dummy)
        assert 0.0 <= score <= 1.0


def test_analyze_symbol_integration(monkeypatch):
    df = _df()
    df_map = {"1h": df}

    async def fake_classify(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_async", fake_classify)
    async def fake_cached(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "detect_patterns", lambda *_a: {})
    monkeypatch.setattr(ma, "route", lambda *_a, **_k: lambda d, cfg=None: (0.5, "long"))

    async def fake_eval(*_a, **_k):
        return [(0.5, "long", None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: 0.1)
    monkeypatch.setattr(pp, "predict_score", lambda _d, model=None: 0.7)

    cfg = {"timeframe": "1h", "ml_price_predictor": {"enabled": True}}

    async def run():
        return await ma.analyze_symbol("AAA", df_map, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["price_score"] == 0.7
