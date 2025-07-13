import pandas as pd
import pytest

from crypto_bot.indicators import cycle_bias
import crypto_bot.signals.signal_scoring as sc


def test_get_cycle_bias_fetches_values(monkeypatch):
    called = []

    def fake_get(url, timeout=5):
        called.append(url)

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"value": 0.2}

        return Resp()

    monkeypatch.setattr(cycle_bias.requests, "get", fake_get)
    cfg = {"mvrv_url": "http://mvrv", "nupl_url": "http://nupl", "sopr_url": "http://sopr"}
    bias = cycle_bias.get_cycle_bias(cfg)
    assert bias == pytest.approx(1.2)
    assert called == ["http://mvrv", "http://nupl", "http://sopr"]


def test_cycle_bias_adjusts_score(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    monkeypatch.setattr(sc, "get_cycle_bias", lambda cfg=None: 1.5)
    cfg = {"cycle_bias": {"enabled": True}}
    score, direction, _ = sc.evaluate(lambda _df: (0.5, "long"), df, cfg)
    assert direction == "long"
    assert score == pytest.approx(0.75)


def test_cycle_bias_disabled(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})
    called = {}

    def fake(cfg=None):
        called["hit"] = True
        return 0.0

    monkeypatch.setattr(sc, "get_cycle_bias", fake)
    score, _, _ = sc.evaluate(lambda _df: (0.5, "long"), df, {"cycle_bias": {"enabled": False}})
    assert score == pytest.approx(0.5)
    assert "hit" not in called
