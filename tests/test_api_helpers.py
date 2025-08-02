import asyncio
import sys
import types

import numpy as np
import pytest

from crypto_bot.solana import api_helpers


class DummyWS:
    async def close(self):
        pass


class DummyResp:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self):
        self.ws_url = None
        self.ws_timeout = None
        self.http_url = None
        self.closed = False

    async def ws_connect(self, url, *, timeout=None):
        self.ws_url = url
        self.ws_timeout = timeout
        return DummyWS()

    def get(self, url, headers=None, timeout=10):
        self.http_url = url
        return DummyResp({"bundle": "ok"})

    async def close(self):
        self.closed = True


def test_helius_ws(monkeypatch):
    session = DummySession()
    monkeypatch.setattr(
        api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session})
    )

    async def _run():
        async with api_helpers.helius_ws("k") as ws:
            assert isinstance(ws, DummyWS)

    asyncio.run(_run())
    assert session.ws_url.endswith("k")
    assert session.ws_timeout == 30
    assert session.closed


def test_helius_ws_connect_failure(monkeypatch):
    class FailSession(DummySession):
        async def ws_connect(self, url, *, timeout=None):
            self.ws_url = url
            self.ws_timeout = timeout
            raise RuntimeError("boom")

    session = FailSession()
    monkeypatch.setattr(
        api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session})
    )

    async def _run():
        async with api_helpers.helius_ws("k"):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(_run())
    assert session.closed


def test_fetch_jito_bundle(monkeypatch):
    session = DummySession()
    monkeypatch.setattr(api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session}))
    arr = np.array([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(api_helpers, "predict_bundle_regime", lambda d: arr)
    data = asyncio.run(api_helpers.fetch_jito_bundle("123", "key"))
    assert np.array_equal(data["predicted_regime"], arr)
    assert session.http_url.endswith("/123")


def test_predict_bundle_regime_missing_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.array_equal(api_helpers.predict_bundle_regime({}), expected)


def test_predict_bundle_regime_model(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    class FakeModel:
        def predict(self, X):
            return np.array([[0.1, 0.2, 0.3, 0.4]])

    class FakeClient:
        pass

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        types.SimpleNamespace(create_client=lambda u, k: FakeClient()),
    )
    monkeypatch.setitem(
        sys.modules,
        "coinTrader_Trainer.ml_trainer",
        types.SimpleNamespace(load_model=lambda name: FakeModel()),
    )

    session = DummySession()
    monkeypatch.setattr(
        api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session})
    )

    probs = api_helpers.predict_bundle_regime({"priority_fee": 1, "tx_count": 2})
    assert np.array_equal(probs, np.array([0.1, 0.2, 0.3, 0.4]))

    monkeypatch.setattr(api_helpers, "predict_bundle_regime", lambda _d: np.array([0.4, 0.3, 0.2, 0.1]))
    data = asyncio.run(api_helpers.fetch_jito_bundle("123", "key"))
    assert np.array_equal(data["predicted_regime"], np.array([0.4, 0.3, 0.2, 0.1]))
    assert session.http_url.endswith("/123")
    assert session.closed


def test_extract_bundle_features():
    bundle = {"tx_count": 2, "priority_fee": 1}
    feats = api_helpers.extract_bundle_features(bundle)
    assert np.array_equal(feats, np.array([1.0, 2.0]))
