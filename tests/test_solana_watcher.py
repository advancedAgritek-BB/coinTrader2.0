import asyncio

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

    def get(self, url, timeout=10):
        return DummyResp(self._data)


def test_watcher_yields_event(monkeypatch):
    data = {
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
    session = DummySession(data)
    monkeypatch.setattr(watcher, "aiohttp", type("M", (), {"ClientSession": lambda: session}))

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
