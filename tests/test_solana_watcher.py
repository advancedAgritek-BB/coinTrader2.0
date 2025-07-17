import asyncio
import logging
import types
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
    assert session.json["method"] == "dex.getNewPools"
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}


def test_env_substitution(monkeypatch):
    monkeypatch.setenv("HELIUS_KEY", "ABC")
    w = PoolWatcher("https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY", interval=0)
    assert w.url.endswith("api-key=ABC")


def test_env_missing(monkeypatch):
    monkeypatch.delenv("HELIUS_KEY", raising=False)
    with pytest.raises(ValueError):
        PoolWatcher("https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY", interval=0)
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
        status = 500

    session = FailingSession([DummyClientError("boom"), data_ok])
    aiohttp_mod = type(
        "M",
        (),
        {
            "ClientSession": lambda: session,
            "ClientError": DummyClientError,
            "ClientResponseError": DummyClientError,
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


def test_watcher_logs_404_and_continues(monkeypatch, caplog):
    data_ok = {"pools": [{"address": "P3"}]}

    class Dummy404(Exception):
        def __init__(self, status=404):
            self.status = status

    session = FailingSession([Dummy404(), data_ok])
    aiohttp_mod = type(
        "M",
        (),
        {
            "ClientSession": lambda: session,
            "ClientError": Exception,
            "ClientResponseError": Dummy404,
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

    with caplog.at_level(logging.ERROR):
        event = asyncio.run(run_once())
    assert event.pool_address == "P3"
    assert any(
        "http://test" in rec.message and "https://mainnet.helius-rpc.com/v1/?api-key=YOUR_KEY" in rec.message
        for rec in caplog.records
    )


def test_watcher_raises_after_consecutive_404(monkeypatch):
    class Dummy404(Exception):
        def __init__(self, status=404):
            self.status = status

    session = FailingSession([Dummy404(), Dummy404()])
    aiohttp_mod = type(
        "M",
        (),
        {
            "ClientSession": lambda: session,
            "ClientError": Exception,
            "ClientResponseError": Dummy404,
        },
    )
    monkeypatch.setattr(watcher, "aiohttp", aiohttp_mod)
    w = PoolWatcher("http://test", interval=0, max_failures=2)

    async def run_once():
        gen = w.watch()
        try:
            await gen.__anext__()
        finally:
            w.stop()
            await gen.aclose()

    with pytest.raises(RuntimeError):
        asyncio.run(run_once())
    assert session.json["method"] == "dex.getNewPools"
    assert session.json["params"] == {"protocols": ["raydium"], "limit": 50}


def test_watch_ws_unauthorized(monkeypatch, caplog):
    class DummyHandshakeError(Exception):
        def __init__(self, status=401):
            self.status = status

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def ws_connect(self, url):
            raise DummyHandshakeError()

    aiohttp_mod = type(
        "M",
        (),
        {
            "ClientSession": lambda: DummySession(),
            "WSServerHandshakeError": DummyHandshakeError,
            "WSMsgType": types.SimpleNamespace(TEXT="text"),
        },
    )
    monkeypatch.setattr(watcher, "aiohttp", aiohttp_mod)

    w = PoolWatcher(
        "http://test",
        interval=0,
        websocket_url="ws://x",
        raydium_program_id="PGM",
    )

    async def run_once():
        gen = w.watch()
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    with caplog.at_level(logging.ERROR):
        asyncio.run(run_once())
    assert not w._running
    assert any(
        "Unauthorized WebSocket connection" in rec.message for rec in caplog.records
    )
