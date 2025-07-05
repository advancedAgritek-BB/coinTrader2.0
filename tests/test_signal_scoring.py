import pandas as pd
import pytest

import crypto_bot.signals.signal_scoring as sc


def dummy_strategy(df):
    return 0.8, "long"


def test_evaluate_blending_half(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    monkeypatch.setattr(sc, "predict_signal", lambda _df: 0.4)
    cfg = {"ml_signal_model": {"enabled": True, "weight": 0.5}}
    score, direction = sc.evaluate(dummy_strategy, df, cfg)
    assert direction == "long"
    assert score == pytest.approx(0.6)


def test_evaluate_blending_custom_weight(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    monkeypatch.setattr(sc, "predict_signal", lambda _df: 0.8)
    cfg = {"ml_signal_model": {"enabled": True, "weight": 0.2}}
    score, _ = sc.evaluate(lambda _df: (0.2, "long"), df, cfg)
    assert score == pytest.approx(0.32)


def test_evaluate_strategies_selects_best(monkeypatch):
    df = pd.DataFrame({"close": [1, 2, 3, 4]})

    def strat_a(df, cfg=None):
        return 0.2, "long"

    def strat_b(df, cfg=None):
        return 0.6, "short"

    calls = []

    def dummy_drawdown(_df, lookback=20):
        calls.append(True)
        return -0.1

    monkeypatch.setattr(sc, "compute_drawdown", dummy_drawdown)
    res = sc.evaluate_strategies([strat_a, strat_b], df, {})
    assert res["name"] == "strat_b"
    assert res["direction"] == "short"
    assert res["score"] == 0.6
    assert calls  # ensure compute_drawdown called

