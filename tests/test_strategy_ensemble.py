import pandas as pd
import asyncio
import numpy as np
import pytest

import crypto_bot.strategy_router as sr
from crypto_bot.strategy_router import RouterConfig
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
    cfg = RouterConfig.from_dict({"signal_fusion": {"fusion_method": "weight"}})
    score, direction = sr.evaluate_regime("trending", df, cfg)
    assert direction == "long"
    assert pytest.approx(score) == 0.65


def test_evaluate_regime_min_conf(monkeypatch):
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]})
    monkeypatch.setattr(sr, "get_strategies_for_regime", lambda r, c=None: [strat_a, strat_b])
    monkeypatch.setattr(
        "crypto_bot.utils.regime_pnl_tracker.compute_weights",
        lambda r: {"strat_a": 0.2, "strat_b": 0.05},
    )
    cfg = RouterConfig.from_dict({"signal_fusion": {"fusion_method": "weight", "min_confidence": 0.1}})
    score, direction = sr.evaluate_regime("trending", df, cfg)
    assert direction == "long"
    assert score == pytest.approx(0.8)


def test_analyze_symbol_ensemble_mode(monkeypatch):
    vals = np.linspace(1, 2, 60)
    df = pd.DataFrame(
        {
            "open": vals,
            "high": vals,
            "low": vals,
            "close": vals,
            "volume": np.ones_like(vals),
        }
    )

    async def fake_async(*_a, **_k):
        return "trending", {"trending": 1.0}

    monkeypatch.setattr("crypto_bot.utils.market_analyzer.classify_regime_async", fake_async)
    async def fake_cached(*_a, **_k):
        return "trending", 1.0

    monkeypatch.setattr("crypto_bot.utils.market_analyzer.classify_regime_cached", fake_cached)
    async def fake_run(df_, strategies, symbol, cfg_, regime=None):
        return [(strat_a, 0.6, "long")]

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "run_candidates", fake_run)
    monkeypatch.setattr(sr, "strategy_for", lambda r, c=None: strat_a)
    monkeypatch.setattr(ma.meta_selector, "_scores_for", lambda r: {})

    async def run():
        cfg = {
            "timeframe": "1h",
            "strategy_evaluation_mode": "ensemble",
            "scoring_weights": {"strategy_score": 1.0},
            "signal_fusion": {"fusion_method": "weight"},
        }
        return await analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    res = asyncio.run(run())
    expected = 0.6 / (1 + np.log1p(df["volume"].iloc[-1]) / 10)
    assert res["score"] == pytest.approx(expected)
    assert res["direction"] == "long"


def test_analyze_symbol_ensemble_default_min_conf(monkeypatch):
    vals = np.linspace(1, 2, 60)
    df = pd.DataFrame(
        {
            "open": vals,
            "high": vals,
            "low": vals,
            "close": vals,
            "volume": np.ones_like(vals),
        }
    )

    async def fake_async(*_a, **_k):
        return "trending", {"trending": 1.0}

    async def fake_cached(*_a, **_k):
        return "trending", 1.0

    import crypto_bot.utils.market_analyzer as ma
    monkeypatch.setattr(ma, "classify_regime_async", fake_async)
    monkeypatch.setattr(ma, "classify_regime_cached", fake_cached)
    monkeypatch.setattr(ma, "detect_patterns", lambda _df: {})
    monkeypatch.setattr(ma, "calc_atr", lambda *_a, **_k: pd.Series([0.0]))

    base_called = []

    def base(df_, cfg=None):
        base_called.append(True)
        return 0.4, "long"

    def extra(df_, cfg=None):
        return 0.3, "long"

    captured = []

    async def fake_run(df_, strategies, symbol, cfg_, regime=None):
        captured.extend(strategies)
        return [(strategies[0], 0.4, "long")]

    monkeypatch.setattr(ma, "run_candidates", fake_run)
    monkeypatch.setattr(sr, "strategy_for", lambda r, c=None: base)
    monkeypatch.setattr(ma, "strategy_for", lambda r, c=None: base)
    monkeypatch.setattr(sr, "get_strategy_by_name", lambda n: {"extra": extra}.get(n))
    monkeypatch.setattr(ma, "get_strategy_by_name", lambda n: {"extra": extra}.get(n))
    monkeypatch.setattr(ma.meta_selector, "_scores_for", lambda r: {"extra": 0.1})

    async def run():
        cfg = {
            "timeframe": "1h",
            "strategy_evaluation_mode": "ensemble",
            "scoring_weights": {"strategy_score": 1.0},
            "signal_fusion": {"fusion_method": "weight"},
        }
        return await analyze_symbol("AAA", {"1h": df}, "cex", cfg, None)

    res = asyncio.run(run())
    assert len(captured) == 1
    assert captured[0] is base
    assert res["direction"] == "long"

