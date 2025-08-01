import json
import types
import pytest

from crypto_bot.solana.watcher import PoolWatcher, NewPoolEvent
import crypto_bot.solana.watcher as watcher_mod


class DummyMsg:
    def __init__(self, data, msg_type="text"):
        self.data = data
        self.type = msg_type


class DummyWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __aiter__(self):
        async def gen():
            for m in self.messages:
                yield DummyMsg(m)
        return gen()


class DummySession:
    def __init__(self, wss):
        self.wss = wss if isinstance(wss, list) else [wss]
        self.calls = 0
        self.url = None

    def ws_connect(self, url):
        self.url = url
        ws = self.wss[self.calls]
        self.calls += 1
        return ws

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class AiohttpMod:
    WSMsgType = types.SimpleNamespace(
        TEXT="text",
        CLOSED="closed",
        ERROR="error",
    )

    def __init__(self, session):
        self._session = session

    def ClientSession(self):
        return self._session


@pytest.mark.asyncio
async def test_parses_raydium_event(monkeypatch):
    msg = json.dumps({
        "method": "logsNotification",
        "params": {"result": {"signature": "sig", "logs": ["initialize2"]}}
    })
    ws = DummyWS([msg])
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(PoolWatcher, "_predict_breakout", lambda self, e: 1.0)

    async def enrich(self, event, sig, sess):
        event.pool_address = "P"
        event.token_mint = "M"
        event.creator = "C"
        event.liquidity = 10.0

    monkeypatch.setattr(PoolWatcher, "_enrich_event", enrich)
    monkeypatch.setattr(watcher_mod, "aiohttp", aiohttp_mod)

    watcher = PoolWatcher("u", 0, "ws://x", "PGM")
    gen = watcher.watch()
    event = await gen.__anext__()
    assert isinstance(event, NewPoolEvent)
    assert event.pool_address == "P"
    assert event.token_mint == "M"
    assert event.creator == "C"
    watcher.stop()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    await gen.aclose()


@pytest.mark.asyncio
async def test_parses_pump_fun(monkeypatch):
    msg = json.dumps({
        "method": "logsNotification",
        "params": {"result": {"signature": "sig", "logs": ["InitializeMint2"]}}
    })
    ws = DummyWS([msg])
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(PoolWatcher, "_predict_breakout", lambda self, e: 1.0)

    async def enrich(self, event, sig, sess):
        event.token_mint = "M"
        event.creator = "C"
        event.liquidity = 20.0

    monkeypatch.setattr(PoolWatcher, "_enrich_event", enrich)
    monkeypatch.setattr(watcher_mod, "aiohttp", aiohttp_mod)

    watcher = PoolWatcher("u", 0, "ws://x", "PGM")
    gen = watcher.watch()
    event = await gen.__anext__()
    assert event.token_mint == "M"
    assert event.creator == "C"
    watcher.stop()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    await gen.aclose()


@pytest.mark.asyncio
async def test_reconnect_on_close(monkeypatch):
    msg_close = DummyMsg(None, "closed")
    msg_ok = json.dumps({
        "method": "logsNotification",
        "params": {"result": {"signature": "sig", "logs": ["initialize2"]}}
    })
    ws1 = DummyWS([msg_close])
    ws2 = DummyWS([msg_ok])
    session = DummySession([ws1, ws2])
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(PoolWatcher, "_predict_breakout", lambda self, e: 1.0)

    async def enrich(self, event, sig, sess):
        event.pool_address = "P"
        event.token_mint = "M"
        event.creator = "C"
        event.liquidity = 10.0

    monkeypatch.setattr(PoolWatcher, "_enrich_event", enrich)
    monkeypatch.setattr(watcher_mod, "aiohttp", aiohttp_mod)

    watcher = PoolWatcher("u", 0, "ws://x", "PGM")
    gen = watcher.watch()
    evt = await gen.__anext__()
    assert evt.pool_address == "P"
    assert session.calls == 2
    watcher.stop()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    await gen.aclose()


class Resp:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyHTTP:
    def __init__(self, payload):
        self.payload = payload
        self.sent = None

    def post(self, url, json=None):
        self.sent = json
        return Resp(self.payload)


@pytest.mark.asyncio
async def test_enrich_event_decimals():
    payload = {
        "result": {
            "meta": {
                "logMessages": ["initialize2"],
                "preTokenBalances": [
                    {"uiTokenAmount": {"uiAmount": 1, "decimals": 6}},
                    {"uiTokenAmount": {"uiAmount": 0, "decimals": 2}},
                ],
                "postTokenBalances": [
                    {"uiTokenAmount": {"uiAmount": 3, "decimals": 6}},
                    {"uiTokenAmount": {"uiAmount": 10, "decimals": 2}},
                ],
            },
            "transaction": {
                "message": {"accountKeys": ["C", "M", "P"]}
            },
            "blockTime": 123,
        }
    }

    session = DummyHTTP(payload)
    watcher = PoolWatcher("u", 0)
    event = NewPoolEvent("", "", "", 0.0)
    await watcher._enrich_event(event, "sig", session)
    assert event.pool_address == "P"
    assert event.token_mint == "M"
    assert event.creator == "C"
    assert event.tx_count == 2
    assert event.liquidity == 2001000
