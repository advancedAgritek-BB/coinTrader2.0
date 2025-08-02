import asyncio
import importlib.util
import types
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pool_ws_monitor",
    Path(__file__).resolve().parents[1] / "crypto_bot/solana/pool_ws_monitor.py",
)
pool_ws_monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_ws_monitor)


@pytest.fixture(autouse=True)
def _mock_listing_date():
    """Override global fixture to avoid heavy imports."""
    yield


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
    WSMsgType = types.SimpleNamespace(
        TEXT="text",
        CLOSED="closed",
        ERROR="error",
    )

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
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    asyncio.run(run())
    assert session.url == "wss://mainnet.helius-rpc.com/?api-key=KEY"
    assert ws.sent and ws.sent[0]["params"][0]["accountInclude"] == ["PGM"]
    assert ws.sent[0]["params"][1]["encoding"] == "jsonParsed"
    assert ws.sent[0]["params"][1]["maxSupportedTransactionVersion"] == 0


def test_yields_transactions(monkeypatch):
    messages = [
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 1,
                    "tx_count": 11,
                    "liquidity": 100,
                    "meta": {
                        "postTokenBalances": [
                            {"uiTokenAmount": {"uiAmount": 100.0}}
                        ]
                    },
                }
            },
        },
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 2,
                    "tx_count": 12,
                    "liquidity": 100,
                    "meta": {
                        "postTokenBalances": [
                            {"uiTokenAmount": {"uiAmount": 100.0}}
                        ]
                    },
                }
            },
        },
    ]
    ws = DummyWS(messages)
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(pool_ws_monitor, "predict_regime", lambda _: "breakout")

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        results = [await gen.__anext__(), await gen.__anext__()]
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return results

    res = asyncio.run(run())
    assert res == [
        {
            "tx": 1,
            "tx_count": 11,
            "liquidity": 100,
            "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 100.0}}]},
            "predicted_regime": "breakout",
        },
        {
            "tx": 2,
            "tx_count": 12,
            "liquidity": 100,
            "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 100.0}}]},
            "predicted_regime": "breakout",
        },
    ]


def test_reconnect_on_close(monkeypatch):
    ws1 = DummyWS([])
    ws2 = DummyWS([
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 3,
                    "tx_count": 15,
                    "liquidity": 100,
                    "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 100.0}}]},
                }
            },
        }
    ])
    session = DummySession([ws1, ws2])
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(pool_ws_monitor, "predict_regime", lambda _: "breakout")
    ws1.messages = [types.SimpleNamespace(type=aiohttp_mod.WSMsgType.CLOSED, data=None)]

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        result = await gen.__anext__()
        return result

    res = asyncio.run(run())
    assert res == {
        "tx": 3,
        "tx_count": 15,
        "liquidity": 100,
        "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 100.0}}]},
        "predicted_regime": "breakout",
    }
    assert session.calls == 2


def test_watch_pool_filters(monkeypatch):
    messages = [
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 1,
                    "tx_count": 5,
                    "liquidity": 40,
                    "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 40.0}}]},
                }
            },
        },
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 2,
                    "tx_count": 12,
                    "liquidity": 60,
                    "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 60.0}}]},
                }
            },
        },
    ]
    ws = DummyWS(messages)
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    monkeypatch.setattr(pool_ws_monitor, "predict_regime", lambda _: "breakout")

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM", min_liquidity=50)
        result = await gen.__anext__()
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return result

    res = asyncio.run(run())
    assert res == {
        "tx": 2,
        "tx_count": 12,
        "liquidity": 60,
        "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 60.0}}]},
        "predicted_regime": "breakout",
    }


def test_ignores_non_breakout(monkeypatch):
    messages = [
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 1,
                    "tx_count": 10,
                    "liquidity": 80,
                    "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 80.0}}]},
                }
            },
        },
        {
            "method": "transactionNotification",
            "params": {
                "result": {
                    "tx": 2,
                    "tx_count": 20,
                    "liquidity": 90,
                    "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 90.0}}]},
                }
            },
        },
    ]
    ws = DummyWS(messages)
    session = DummySession(ws)
    aiohttp_mod = AiohttpMod(session)
    monkeypatch.setattr(pool_ws_monitor, "aiohttp", aiohttp_mod)
    labels = iter(["trending", "breakout"])
    monkeypatch.setattr(pool_ws_monitor, "predict_regime", lambda _: next(labels))

    async def run():
        gen = pool_ws_monitor.watch_pool("KEY", "PGM")
        result = await gen.__anext__()
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        return result

    res = asyncio.run(run())
    assert res == {
        "tx": 2,
        "tx_count": 20,
        "liquidity": 90,
        "meta": {"postTokenBalances": [{"uiTokenAmount": {"uiAmount": 90.0}}]},
        "predicted_regime": "breakout",
    }
