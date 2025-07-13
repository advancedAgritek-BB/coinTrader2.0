import asyncio
import pytest

from crypto_bot.solana import watcher
from crypto_bot.solana.watcher import PoolWatcher, NewPoolEvent


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json=None, timeout=10):
        self.json = json
        return DummyResp(self._data)


class FailingSession(DummySession):
    def __init__(self, responses):
        # responses is a list of either dict or Exception
        self._responses = responses
        self.calls = 0

    def post(self, url, json=None, timeout=10):
        self.json = json
        resp = self._responses[self.calls]
        self.calls += 1
        if isinstance(resp, Exception):
            raise resp
        return DummyResp(resp)


def test_watcher_yields_event(monkeypatch):
    data = {
        "result": {
            "pools": [
                {
                    "address": "P1",
                    "tokenMint": "M1",
                    "creator": "C1",
                    "liquidity": 10.5,
                    "txCount": 3,
                }
            ]
        }
    }
    session = DummySession(data)
    monkeypatch.setattr(
        watcher,
        "aiohttp",
        type("M", (), {"ClientSession": lambda: session}),
    )

    w = PoolWatcher("http://test", interval=0)

    async def run_once():
        gen = w.watch()
        event = await gen.__anext__()
        w.stop()
        await gen.aclose()
        return event

    event = asyncio.run(run_once())
    assert isinstance(event, NewPoolEvent)
    assert event.pool_address == "P1"
    assert event.token_mint == "M1"
    assert event.creator == "C1"
    assert event.liquidity == 10.5
    assert event.tx_count == 3
    assert session.json["method"] == "getPools"


def test_env_substitution(monkeypatch):
    monkeypatch.setenv("HELIUS_KEY", "ABC")
    w = PoolWatcher("https://rpc.helius.xyz/?api-key=YOUR_KEY", interval=0)
    assert w.url.endswith("api-key=ABC")


def test_env_missing(monkeypatch):
    monkeypatch.delenv("HELIUS_KEY", raising=False)
    with pytest.raises(ValueError):
        PoolWatcher("https://rpc.helius.xyz/?api-key=YOUR_KEY", interval=0)
def test_watcher_continues_after_error(monkeypatch):
    data_ok = {
        "result": {
            "pools": [
                {
                    "address": "P2",
                    "tokenMint": "M2",
                    "creator": "C2",
                    "liquidity": 1.0,
                    "txCount": 1,
                }
            ]
        }
    }
    class DummyClientError(Exception):
        pass

    session = FailingSession([DummyClientError("boom"), data_ok])
    aiohttp_mod = type(
        "M",
        (),
        {
            "ClientSession": lambda: session,
            "ClientError": DummyClientError,
        },
    )
    monkeypatch.setattr(watcher, "aiohttp", aiohttp_mod)

    w = PoolWatcher("http://test", interval=0)

    async def run_once():
        gen = w.watch()
        event = await gen.__anext__()
        w.stop()
        await gen.aclose()
        return event

    event = asyncio.run(run_once())
    assert event.pool_address == "P2"
    assert event.token_mint == "M2"
    assert event.creator == "C2"
    assert session.json["method"] == "getPools"
