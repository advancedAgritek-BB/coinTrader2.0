import asyncio
import pytest

from crypto_bot.utils.market_loader import load_kraken_symbols
import crypto_bot.utils.market_loader as ml_mod
from crypto_bot.utils import symbol_utils


REAL_SLEEP = asyncio.sleep


class SlowExchange:
    def __init__(self):
        self.calls = 0
        self.has = {}

    async def load_markets(self):
        self.calls += 1
        await REAL_SLEEP(0.2)
        return {
            "BTC/USD": {
                "active": True,
                "type": "spot",
                "base": "BTC",
                "quote": "USD",
            }
        }


class SlowTypeExchange:
    def __init__(self):
        self.calls = 0
        self.has = {"fetchMarketsByType": True}

    async def fetch_markets_by_type(self, m_type):
        self.calls += 1
        await REAL_SLEEP(0.2)
        return []


@pytest.mark.asyncio
async def test_load_kraken_symbols_timeout_retry(monkeypatch):
    async def fast_sleep(_):
        return

    monkeypatch.setattr(ml_mod.asyncio, "sleep", fast_sleep)
    ex = SlowExchange()
    cfg = {"symbol_scan_timeout": 0.01}
    result = await load_kraken_symbols(ex, config=cfg)
    assert result is None
    assert ex.calls > 1


@pytest.mark.asyncio
async def test_load_kraken_symbols_timeout_fetch_by_type(monkeypatch):
    async def fast_sleep(_):
        return

    monkeypatch.setattr(ml_mod.asyncio, "sleep", fast_sleep)
    ex = SlowTypeExchange()
    cfg = {"symbol_scan_timeout": 0.01, "exchange_market_types": ["spot"]}
    result = await load_kraken_symbols(ex, config=cfg)
    assert result is None
    assert ex.calls > 1


def test_get_filtered_symbols_passes_timeout(monkeypatch):
    class DummyExchange:
        def __init__(self):
            self.t = None

        def list_markets(self, timeout):
            self.t = timeout
            return {"BTC/USD": {"quote": "USD", "quoteVolume": 10}}

    async def fake_filter(exchange, symbols, config):
        return [(s, 0.0) for s in symbols], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter)
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0
    ex = DummyExchange()
    cfg = {
        "symbol_refresh_minutes": 0,
        "symbol_filter": {},
        "symbols": ["BTC/USD"],
        "symbol_scan_timeout": 12,
    }
    asyncio.run(symbol_utils.get_filtered_symbols(ex, cfg))
    assert ex.t == 12
