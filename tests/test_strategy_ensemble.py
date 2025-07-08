import pandas as pd
import asyncio
import pytest

import crypto_bot.strategy_router as sr
from crypto_bot.utils.market_analyzer import analyze_symbol


def strat_a(df, cfg=None):
    return 0.8, "long"


def strat_b(df, cfg=None):
    return 0.2, "long"


def test_evaluate_regime_fuses_scores(monkeypatch):
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    monkeypatch.setattr(sr, "get_strategies_for_regime", lambda r, c=None: [strat_a, strat_b])
    monkeypatch.setattr(
        "crypto_bot.utils.regime_pnl_tracker.compute_weights",
        lambda r: {"strat_a": 0.75, "strat_b": 0.25},
    )
    score, direction = sr.evaluate_regime("trending", df, {"signal_fusion": {"fusion_method": "weight"}})
    assert direction == "long"
    assert pytest.approx(score) == 0.65


def test_evaluate_regime_min_conf(monkeypatch):
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    monkeypatch.setattr(sr, "get_strategies_for_regime", lambda r, c=None: [strat_a, strat_b])
    monkeypatch.setattr(
        "crypto_bot.utils.regime_pnl_tracker.compute_weights",
        lambda r: {"strat_a": 0.2, "strat_b": 0.05},
    )
    cfg = {"signal_fusion": {"fusion_method": "weight", "min_confidence": 0.1}}
    score, direction = sr.evaluate_regime("trending", df, cfg)
    assert direction == "long"
    assert score == pytest.approx(0.8)


def test_analyze_symbol_ensemble_mode(monkeypatch):
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})

    async def fake_async(*_a, **_k):
        return "trending", {"trending": 1.0}

    monkeypatch.setattr("crypto_bot.utils.market_analyzer.classify_regime_async", fake_async)
    async def fake_cached(*_a, **_k):
        return "trending", 1.0

    monkeypatch.setattr("crypto_bot.utils.market_analyzer.classify_regime_cached", fake_cached)
    monkeypatch.setattr(sr, "evaluate_regime", lambda r, d, cfg=None: (0.6, "long"))

    async def run():
        cfg = {
            "timeframe": "1h",
            "strategy_evaluation_mode": "ensemble",
            "scoring_weights": {"strategy_score": 1.0},
            "signal_fusion": {"fusion_method": "weight"},
        }
        return await analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    res = asyncio.run(run())
    assert res["score"] == 0.6
    assert res["direction"] == "long"
