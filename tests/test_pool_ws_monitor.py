import asyncio
import types
import pytest

from crypto_bot.solana import pool_ws_monitor


class DummyMsg:
    def __init__(self, data):
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
        self.ws = ws
        self.url = None

    def ws_connect(self, url):
        self.url = url
        return self.ws

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class AiohttpMod:
    WSMsgType = types.SimpleNamespace(TEXT="text", CLOSED="closed", ERROR="error")

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
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

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
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        results = [await gen.__anext__(), await gen.__anext__()]
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return results

    res = asyncio.run(run())
    assert res == [{"tx": 1}, {"tx": 2}]
