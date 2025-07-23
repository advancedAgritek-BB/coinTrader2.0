import asyncio
import pandas as pd
from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main


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
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, timeout=10):
        self.calls.append(url)
        if "price_feeds?query=BTC" in url:
            return DummyResp(
                [{"id": "ID", "attributes": {"base": "BTC", "quote_currency": "USD"}}]
            )
        return DummyResp([{"price": {"price": "100", "expo": 0}}])


def test_enrich_with_pyth(monkeypatch):
    df1 = pd.DataFrame({"close": [1.0]})
    df2 = pd.DataFrame({"close": [1.0]})
    cache = {"BTC/USDC": df1, "BTC/USD": df2}
    ctx = BotContext(
        positions={},
        df_cache={"1h": cache},
        regime_cache={},
        config={"pyth_quotes": ["USDC", "USD"]},
    )
    ctx.current_batch = ["BTC/USDC", "BTC/USD"]
    monkeypatch.setattr(main.aiohttp, "ClientSession", lambda: DummySession())
    asyncio.run(main.enrich_with_pyth(ctx))
    assert df1["close"].iloc[-1] == 100.0
    assert df2["close"].iloc[-1] == 100.0
