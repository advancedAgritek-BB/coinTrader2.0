import asyncio
import pytest

from crypto_bot.solana import token_utils
import importlib


class DummyResp:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, data):
        self._data = data
        self.json_payload = None
        self.url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json=None, timeout=10):
        self.url = url
        self.json_payload = json
        return DummyResp(self._data)

    def get(self, url, timeout=10):
        self.url = url
        return DummyResp(self._data)


class FailingSession(DummySession):
    def __init__(self, exc):
        super().__init__({})
        self.exc = exc

    def post(self, url, json=None, timeout=10):
        raise self.exc


def test_get_token_accounts(monkeypatch):
    data = {
        "result": {
            "value": [
                {
                    "account": {
                        "data": {
                            "parsed": {
                                "info": {
                                    "tokenAmount": {"uiAmount": 1},
                                    "mint": "A",
                                }
                            }
                        }
                    }
                },
                {
                    "account": {
                        "data": {
                            "parsed": {
                                "info": {
                                    "tokenAmount": {"uiAmount": 1},
                                    "mint": "B",
                                }
                            }
                        }
                    }
                },
            ]
        }
    }
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session, "ClientError": Exception})
    monkeypatch.setenv("HELIUS_KEY", "k")
    monkeypatch.setenv("MIN_BALANCE_THRESHOLD", "0.001")
    importlib.reload(token_utils)
    monkeypatch.setattr(token_utils, "aiohttp", aiohttp_mod)

    async def fake_enrich(pubkey, session):
        return {"mint": pubkey}

    def fake_predict(meta):
        return {"A": 0.6, "B": 0.4}[meta["mint"]]

    monkeypatch.setattr(token_utils, "enrich_with_metadata", fake_enrich)
    monkeypatch.setattr(token_utils, "predict_token_regime", fake_predict)

    accounts = asyncio.run(token_utils.get_token_accounts("wallet"))
    assert len(accounts) == 1
    assert accounts[0]["metadata"] == {"mint": "A"}
    assert accounts[0]["ml_score"] == 0.6
    assert session.json_payload["method"] == "getTokenAccountsByOwner"


def test_get_token_accounts_error(monkeypatch):
    class DummyErr(Exception):
        pass

    session = FailingSession(DummyErr("boom"))
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session, "ClientError": DummyErr})
    monkeypatch.setenv("HELIUS_KEY", "k")
    importlib.reload(token_utils)
    monkeypatch.setattr(token_utils, "aiohttp", aiohttp_mod)

    with pytest.raises(DummyErr):
        asyncio.run(token_utils.get_token_accounts("wallet"))
