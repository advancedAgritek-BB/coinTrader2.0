import pandas as pd
import pytest

from crypto_bot.signals import signal_scoring as sc


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

