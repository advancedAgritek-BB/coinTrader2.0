import asyncio
import logging

import pandas as pd
import pytest

from crypto_bot.utils import market_loader


class DummyExchange:
    pass


def test_batch_worker_logs_and_continues(monkeypatch, caplog):
    async def failing_inner(exchange, cache, symbols, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(market_loader, "_update_ohlcv_cache_inner", failing_inner)

    cache: dict[str, pd.DataFrame] = {}
    with caplog.at_level(logging.ERROR):
        res = asyncio.run(
            market_loader.update_ohlcv_cache(
                DummyExchange(), cache, ["BTC/USD"], limit=1
            )
        )
    assert res == cache
    assert any(
        "OHLCV worker: failed" in r.getMessage() for r in caplog.records
    )
