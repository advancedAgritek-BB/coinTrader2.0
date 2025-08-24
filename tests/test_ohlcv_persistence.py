import asyncio
import pandas as pd
import time

from crypto_bot.utils import market_loader

timeframe_seconds = market_loader.timeframe_seconds
update_ohlcv_cache = market_loader.update_ohlcv_cache


class DummyExchange:
    has = {"fetchOHLCV": True}
    timeframes = {"1h": "1h"}
    markets = {"BTC/USD": {"base": "BTC", "quote": "USD", "symbol": "BTC/USD"}}

    def __init__(self):
        self.calls = []

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100, **kwargs):
        self.calls.append({"since": since, "limit": limit})
        tf_sec = timeframe_seconds(None, timeframe)
        start = since or 0
        rows = []
        for i in range(limit):
            ts = start + i * tf_sec * 1000
            rows.append([ts, 1, 1, 1, 1, 1])
        return rows


def test_ohlcv_persistence(tmp_path, monkeypatch):
    async def _listing(_sym: str):
        return 0

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", _listing)
    symbol = "BTC/USD"
    timeframe = "1h"
    max_bootstrap = 10
    tail = 3
    new = 2

    ex = DummyExchange()
    config = {"tail_overlap_bars": tail, "storage_path": str(tmp_path)}
    cache: dict[str, pd.DataFrame] = {}
    tf_sec = timeframe_seconds(None, timeframe)
    now = int(time.time())
    start_since = (now // tf_sec - max_bootstrap) * tf_sec * 1000
    cache = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            [symbol],
            timeframe=timeframe,
            limit=max_bootstrap,
            start_since=start_since,
            config=config,
        )
    )
    assert ex.calls[0]["limit"] >= max_bootstrap

    last_ts_ms = cache[symbol]["timestamp"].iloc[-1] * 1000

    ex.calls = []
    monkeypatch.setattr(
        market_loader,
        "utc_now_ms",
        lambda: last_ts_ms + (new - 1) * tf_sec * 1000,
    )
    cache = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            [symbol],
            timeframe=timeframe,
            limit=new,
            start_since=last_ts_ms - tail * tf_sec * 1000,
            config=config,
        )
    )
    assert len(ex.calls) == 1
    call = ex.calls[0]
    tf_sec = timeframe_seconds(None, timeframe)
    assert call["limit"] == tail + new
    assert call["since"] == last_ts_ms - tail * tf_sec * 1000
