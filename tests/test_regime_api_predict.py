import numpy as np
import pandas as pd

from crypto_bot.regime import api


class DummyModel:
    def predict_proba(self, _df):
        return np.array([[0.1, 0.2, 0.7]])


def test_predict_uses_supabase_model(monkeypatch):
    calls = []

    def fake_load_latest(symbol):
        calls.append(symbol)
        return b"blob", {"label_order": [-1, 0, 1], "feature_list": []}

    monkeypatch.setattr(api, "load_latest_regime", fake_load_latest)
    monkeypatch.setattr(api, "_load_model_from_bytes", lambda blob: DummyModel())
    monkeypatch.setenv("SUPABASE_URL", "http://example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")

    df = pd.DataFrame({"close": [1, 2, 3]})
    pred = api.predict(df)

    assert calls == ["BTCUSDT"]
    assert pred.action == "long"
    assert pred.meta == {"label_order": [-1, 0, 1], "feature_list": []}


def test_predict_falls_back_without_creds(monkeypatch):
    calls = []

    def fake_load_latest(symbol):
        calls.append(symbol)
        return b"", {}

    monkeypatch.setattr(api, "load_latest_regime", fake_load_latest)
    for var in [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_KEY",
        "SUPABASE_API_KEY",
        "SUPABASE_ANON_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    df = pd.DataFrame({"close": [1, 2, 3]})
    pred = api.predict(df)

    assert calls == []  # load_latest_regime not invoked
    assert pred.meta["source"] == "fallback"

