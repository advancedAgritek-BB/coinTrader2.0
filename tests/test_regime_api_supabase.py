"""Tests for Supabase-backed model loading in regime API."""

import numpy as np
import pandas as pd
from crypto_bot.regime import api


class _DummyModel:
    def predict_proba(self, _):
        return np.array([[0.1, 0.2, 0.7]])


class _SupabaseRegistry:
    def __init__(self):
        self.pointer_called = False
        self.latest_called = False

    def load_pointer(self, prefix):
        self.pointer_called = True
        return {"label_order": [-1, 0, 1]}

    def load_latest(self, prefix, allow_fallback=False):
        self.latest_called = True
        return b"model"


def test_predict_uses_supabase_model_when_credentials_present(monkeypatch):
    reg = _SupabaseRegistry()
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setattr(api, "_registry", reg)
    monkeypatch.setattr(api, "_load_model_from_bytes", lambda b: _DummyModel())
    features = pd.DataFrame({"close": [1, 2, 3]})

    pred = api.predict(features)

    assert reg.pointer_called and reg.latest_called
    assert pred.meta["source"] == "registry"
    assert pred.action == "long"


class _FailingRegistry:
    def load_pointer(self, prefix):  # pragma: no cover - guard path
        raise RuntimeError("no supabase")


def test_predict_falls_back_without_supabase(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(api, "_registry", _FailingRegistry())
    features = pd.DataFrame({"close": [1, 2, 3]})

    pred = api.predict(features)

    assert pred.meta["source"] == "fallback"
