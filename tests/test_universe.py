import pytest

from crypto_bot.universe import build_tradable_set
from crypto_bot.main import initial_scan, SessionState


class DummyExchange:
    def list_markets(self):
        return {
            "AAA/USD": {"quote": "USD", "status": "online"},
            "BBB/USD": {"quote": "USD", "status": "online"},
            "CCC/USDT": {"quote": "USDT", "status": "online"},
        }

    async def fetch_tickers(self):
        return {
            "AAA/USD": {"bid": 1.0, "ask": 1.01, "quoteVolume": 1000},
            # Fails spread
            "BBB/USD": {"bid": 1.0, "ask": 1.3, "quoteVolume": 5000},
            # Would pass but is blacklisted
            "CCC/USDT": {"bid": 1.0, "ask": 1.02, "quoteVolume": 2000},
        }


@pytest.mark.asyncio
async def test_build_tradable_set_filters_and_limits():
    ex = DummyExchange()
    res = await build_tradable_set(
        ex,
        allowed_quotes=["USD", "USDT"],
        min_daily_volume_quote=800,
        max_spread_pct=2.0,
        blacklist=["CCC/USDT"],
        max_pairs=1,
    )
    assert res == ["AAA/USD"]


@pytest.mark.asyncio
async def test_initial_scan_uses_tradable_symbols(monkeypatch):
    batches: list[list[str]] = []

    async def fake_update_multi(exchange, cache, batch, cfg, **kwargs):
        batches.append(list(batch))
        return {}

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr(
        "crypto_bot.main.update_multi_tf_ohlcv_cache", fake_update_multi
    )
    monkeypatch.setattr(
        "crypto_bot.main.update_regime_tf_cache", fake_update_regime
    )

    cfg = {
        "symbols": ["AAA/USD", "BBB/USD"],
        "tradable_symbols": ["AAA/USD"],
        "timeframes": ["1h"],
        "scan_lookback_limit": 50,
    }
    await initial_scan(DummyExchange(), cfg, SessionState())
    assert batches and batches[0] == ["AAA/USD"]


class BroadExchange:
    def list_markets(self):
        markets = {}
        for i in range(70):
            if i < 40:
                sym = f"AAA{i}/USD"
                markets[sym] = {"quote": "USD", "status": "online"}
            else:
                sym = f"BBB{i}/USDT"
                markets[sym] = {"quote": "USDT", "status": "online"}
        return markets

    async def fetch_tickers(self):
        tickers = {}
        for i in range(70):
            if i < 40:
                sym = f"AAA{i}/USD"
            else:
                sym = f"BBB{i}/USDT"
            tickers[sym] = {"bid": 1.0, "ask": 1.01, "quoteVolume": 1000}
        return tickers


@pytest.mark.asyncio
async def test_build_tradable_set_retains_breadth():
    ex = BroadExchange()
    res = await build_tradable_set(
        ex,
        allowed_quotes=["USD", "USDT"],
        min_daily_volume_quote=500,
        max_spread_pct=2.0,
    )
    assert len(res) == 70


@pytest.mark.asyncio
async def test_build_tradable_set_total_cap():
    ex = BroadExchange()
    res = await build_tradable_set(
        ex,
        allowed_quotes=["USD", "USDT"],
        min_daily_volume_quote=500,
        max_spread_pct=2.0,
        max_pairs=5,
        max_pairs_total=7,
    )
    assert len(res) == 7
