import asyncio
import time
import pandas as pd

from crypto_bot.utils import market_loader


def test_update_multi_tf_ohlcv_cache_clamps(monkeypatch):
    captured: dict[str, int | None] = {}

    async def fake_update(exchange, tf_cache, symbols, timeframe, limit, start_since, **kwargs):
        captured["limit"] = limit
        captured["start_since"] = start_since
        for s in symbols:
            tf_cache[s] = pd.DataFrame(
                [[0, 0, 0, 0, 0, 0]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return tf_cache

    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader, "get_kraken_listing_date", lambda _s: 0)

    class Ex:
        id = "dummy"
        timeframes = {"1m": "1m"}
        symbols = ["BTC/USD"]

    cfg = {
        "timeframes": ["1m"],
        "timeframe_backfill_days": {"1m": 2},
        "warmup_candles": {"1m": 1000},
    }
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            Ex(),
            {},
            ["BTC/USD"],
            cfg,
            limit=5000,
            start_since=0,
        )
    )
    assert captured["limit"] == 1000
    assert captured["start_since"] is None
