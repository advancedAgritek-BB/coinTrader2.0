import asyncio
import numpy as np
import pandas as pd
import pytest

from crypto_bot.utils.market_analyzer import analyze_symbol


def make_df(rows: int = 50) -> pd.DataFrame:
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


def test_strategy_regime_strength(monkeypatch):
    df = make_df()

    class Match:
        @staticmethod
        def matches(r):
            return r == "trending"

    class NoMatch:
        @staticmethod
        def matches(r):
            return False

    def strat_match(df_, cfg=None):
        return 0.5, "long"

    strat_match.regime_filter = Match()

    def strat_nomatch(df_, cfg=None):
        return 0.5, "long"

    strat_nomatch.regime_filter = NoMatch()

    import crypto_bot.utils.market_analyzer as ma

    async def fake_async(*_a, **_k):
        return "trending", {"trending": 1.0}

    async def fake_cached(*_a, **_k):
        return "trending", 1.0

    async def fake_patterns(*_a, **_k):
        return "trending", {}

    monkeypatch.setattr(ma, "classify_regime_async", fake_async)
    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "classify_regime_with_patterns_async", fake_patterns)

    async def fake_eval(*_a, **_k):
        return [(0.5, "long", None)]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma, "detect_patterns", lambda _df: {})
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: pd.Series([0.0]))

    cfg = {
        "timeframe": "1h",
        "scoring_weights": {"strategy_score": 0.0, "strategy_regime_strength": 1.0},
    }
    df_map = {"1h": df}

    # Matching regime
    monkeypatch.setattr(ma, "route", lambda *a, **k: strat_match)
    res = asyncio.run(analyze_symbol("AAA", df_map, "cex", cfg, None))
    expected = 1.0 / (1 + np.log1p(df["volume"].iloc[-1]) / 10)
    assert res["score"] == pytest.approx(expected)

    # Non-matching regime
    monkeypatch.setattr(ma, "route", lambda *a, **k: strat_nomatch)
    res = asyncio.run(analyze_symbol("AAA", df_map, "cex", cfg, None))
    assert res["score"] == pytest.approx(0.0)
