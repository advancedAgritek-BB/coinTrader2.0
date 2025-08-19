import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from crypto_bot.utils.market_loader import _fetch_ohlcv_async_inner


class DummyExchange:
    """Minimal exchange mock recording fetch_ohlcv calls."""

    def __init__(self):
        self.calls: list[dict] = []
        self.has = {"fetchOHLCV": True}
        self.timeframes = {"1m": 60}

    async def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None):
        self.calls.append({"since": since, "limit": limit})
        start = since or 0
        tf_ms = 60_000
        return [
            [start + i * tf_ms, 0, 0, 0, 0, 0]
            for i in range(limit)
        ]


@pytest.mark.asyncio
async def test_planner_requests_expected_candles():
    """Planner should fetch >= 3 days of 1m candles across pages."""
    exch = DummyExchange()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since = now_ms - 3 * 24 * 60 * 60 * 1000
    needed = 3 * 1440
    data = await _fetch_ohlcv_async_inner(
        exch, "BTC/USD", timeframe="1m", limit=needed, since=since
    )
    assert len(data) >= needed
    assert len(exch.calls) > 1  # multiple pages used


@pytest.mark.asyncio
async def test_planner_since_advances_per_page():
    """Each page should advance since by exactly one candle."""
    exch = DummyExchange()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since = now_ms - 3 * 24 * 60 * 60 * 1000
    needed = 3 * 1440
    await _fetch_ohlcv_async_inner(
        exch, "BTC/USD", timeframe="1m", limit=needed, since=since
    )

    assert exch.calls[0]["since"] == since
    tf_ms = 60_000
    for prev, cur in zip(exch.calls, exch.calls[1:]):
        expected = prev["since"] + prev["limit"] * tf_ms
        assert cur["since"] == expected
