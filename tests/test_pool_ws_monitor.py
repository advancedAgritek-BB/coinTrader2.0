import asyncio
import types
import pytest

from crypto_bot.solana import pool_ws_monitor


class DummyMsg:
    def __init__(self, data):
        if hasattr(data, "type"):
            self.type = data.type
            self.data = data.data
        else:
            self.data = data
            self.type = pool_ws_monitor.aiohttp.WSMsgType.TEXT

    def json(self):
        return self.data


class DummyWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self):
        pass

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
    def __init__(self, ws):
        if isinstance(ws, list):
            self.ws_list = ws
        else:
            self.ws_list = [ws]
        self.calls = 0
        self.url = None

    def ws_connect(self, url):
        self.url = url
        ws = self.ws_list[self.calls]
        self.calls += 1
        return ws

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class AiohttpMod:
    WSMsgType = types.SimpleNamespace(TEXT="text", CLOSED="closed", ERROR="error")

    WSServerHandshakeError = Exception

    def __init__(self, session):
        self._session = session

    def ClientSession(self):
        return self._session


def test_subscription_message(monkeypatch):
    ws = DummyWS([])
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)

    async def run():
        gen = pool_ws_monitor.watch_pool("PGM", api_key="KEY")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), 0.01)

    asyncio.run(run())
    assert session.url == "wss://atlas-mainnet.helius-rpc.com/?api-key=KEY"
    assert ws.sent and ws.sent[0]["params"][0]["accountInclude"] == ["PGM"]


def test_yields_transactions(monkeypatch):
    messages = [
        {"params": {"result": {"tx": 1}}},
        {"params": {"result": {"tx": 2}}},
    ]
    ws = DummyWS(messages)
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)

    async def run():
        gen = pool_ws_monitor.watch_pool("PGM", api_key="KEY")
        results = [await gen.__anext__(), await gen.__anext__()]
        return results

    res = asyncio.run(run())
    assert res == [{"tx": 1}, {"tx": 2}]


def test_reconnect_on_close(monkeypatch):
    ws1 = DummyWS([])
    ws2 = DummyWS([{"params": {"result": {"tx": 3}}}])
    session = DummySession([ws1, ws2])
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    ws1.messages = [types.SimpleNamespace(type=aiohttp_mod.WSMsgType.CLOSED, data=None)]

    async def run():
        gen = pool_ws_monitor.watch_pool("PGM", api_key="KEY")
        result = await gen.__anext__()
        return result

    res = asyncio.run(run())
    assert res == {"tx": 3}
    assert session.calls == 2


def test_watch_pool_env(monkeypatch):
    ws = DummyWS([])
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    monkeypatch.setenv("HELIUS_KEY", "ENVK")

    async def run():
        gen = pool_ws_monitor.watch_pool("PGM")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), 0.01)

    asyncio.run(run())
    assert session.url == "wss://atlas-mainnet.helius-rpc.com/?api-key=ENVK"


def test_watch_pool_missing_key(monkeypatch):
    monkeypatch.delenv("HELIUS_KEY", raising=False)

    async def run():
        gen = pool_ws_monitor.watch_pool("PGM")
        await gen.__anext__()

    with pytest.raises(RuntimeError):
        asyncio.run(run())
