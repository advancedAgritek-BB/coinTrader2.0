import pytest

from crypto_bot.exchange.kraken_client import KrakenClient as AsyncKrakenClient
from crypto_bot.execution.kraken_client import KrakenClient as SyncKrakenClient
from crypto_bot.universe import build_tradable_set


class StubAsyncExchange:
    async def load_markets(self):
        return {"AAA/USD": {"quote": "USD", "status": "online"}}

    async def fetch_tickers(self):
        return {"AAA/USD": {"bid": 1.0, "ask": 1.01, "quoteVolume": 1000}}

    async def close(self):
        pass


class StubSyncExchange:
    def load_markets(self):
        return {"AAA/USD": {"quote": "USD", "status": "online"}}

    def fetch_tickers(self):
        return {"AAA/USD": {"bid": 1.0, "ask": 1.01, "quoteVolume": 1000}}


@pytest.mark.asyncio
async def test_async_kraken_client_list_markets_and_build_tradable_set():
    client = AsyncKrakenClient(StubAsyncExchange())
    markets = await client.list_markets()
    assert isinstance(markets, dict) and markets
    symbols = await build_tradable_set(
        client,
        allowed_quotes=["USD"],
        min_daily_volume_quote=500,
        max_spread_pct=2.0,
    )
    assert symbols == ["AAA/USD"]


@pytest.mark.asyncio
async def test_sync_kraken_client_list_markets_and_build_tradable_set():
    client = SyncKrakenClient(StubSyncExchange())
    markets = client.list_markets()
    assert isinstance(markets, dict) and markets
    symbols = await build_tradable_set(
        client,
        allowed_quotes=["USD"],
        min_daily_volume_quote=500,
        max_spread_pct=2.0,
    )
    assert symbols == ["AAA/USD"]
