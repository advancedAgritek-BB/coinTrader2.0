import numpy as np
import pandas as pd

import logging
import sys
import types

from crypto_bot.regime import api
from crypto_bot.regime import registry


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
    monkeypatch.delenv("CT_SYMBOL", raising=False)

    df = pd.DataFrame({"close": [1, 2, 3]})
    pred = api.predict(df)

    assert calls == ["XRPUSD"]
    assert pred.action == "long"
    assert pred.meta == {"label_order": [-1, 0, 1], "feature_list": []}


def test_predict_allows_symbol_override(monkeypatch):
    calls = []

    def fake_load_latest(symbol):
        calls.append(symbol)
        return b"blob", {"label_order": [-1, 0, 1], "feature_list": []}

    monkeypatch.setattr(api, "load_latest_regime", fake_load_latest)
    monkeypatch.setattr(api, "_load_model_from_bytes", lambda blob: DummyModel())
    monkeypatch.setenv("SUPABASE_URL", "http://example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
    monkeypatch.setenv("CT_SYMBOL", "ETHUSD")

    df = pd.DataFrame({"close": [1, 2, 3]})
    pred = api.predict(df, symbol="BTCUSDT")

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


def test_missing_model_logs_once(monkeypatch, caplog):
    class Response:
        status_code = 404

    class NotFound(Exception):
        def __init__(self):
            self.response = Response()

    class Bucket:
        def download(self, _key):
            raise NotFound()

    class Storage:
        def from_(self, _bucket):
            return Bucket()

    class Client:
        storage = Storage()

    def create_client(url, key):  # noqa: D401 - stub
        return Client()

    monkeypatch.setitem(sys.modules, "supabase", types.SimpleNamespace(create_client=create_client))
    monkeypatch.setenv("SUPABASE_URL", "http://example")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
    monkeypatch.setattr(registry, "_load_fallback", lambda: object())
    monkeypatch.setattr(registry, "_no_model_logged", False)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")
    df = pd.DataFrame({"close": [1, 2, 3]})

    pred1 = api.predict(df)
    pred2 = api.predict(df)

    logs = [r for r in caplog.records if "No regime model found" in r.message]
    assert len(logs) == 1
    assert logs[0].levelno == logging.INFO
    assert pred1.meta["source"] == "fallback"
    assert pred2.meta["source"] == "fallback"

