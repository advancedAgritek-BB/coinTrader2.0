import asyncio
import logging
from types import SimpleNamespace

import pandas as pd

from crypto_bot.utils import market_loader
from crypto_bot.strategy import registry


class Ex:
    id = "dummy"
    timeframes = {"1m": "1m"}
    symbols = ["BTC/USD"]


async def _fake_update(exchange, tf_cache, symbols, timeframe, limit, start_since, **kwargs):
    for s in symbols:
        tf_cache[s] = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0]],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
    return tf_cache


async def _run_update(cfg):
    market_loader.update_ohlcv_cache = _fake_update
    market_loader.get_kraken_listing_date = lambda _s: 0
    await market_loader.update_multi_tf_ohlcv_cache(
        Ex(), {}, ["BTC/USD"], cfg, limit=100, start_since=0
    )


def test_auto_raise_warmup(monkeypatch, caplog):
    stub = SimpleNamespace(__name__="stub", required_lookback=lambda: {"1m": 1440})
    monkeypatch.setattr(registry, "load_from_config", lambda cfg: [stub])
    cfg = {
        "timeframes": ["1m"],
        "warmup_candles": {"1m": 1000},
        "data": {"auto_raise_warmup": True},
    }
    caplog.set_level(logging.INFO)
    asyncio.run(_run_update(cfg))
    assert cfg["warmup_candles"]["1m"] == 1440
    assert "Auto-raising warmup_candles[1m]" in caplog.text


def test_registry_disables_strategy(caplog):
    stub = SimpleNamespace(__name__="stub", required_lookback=lambda: {"1m": 1440})
    cfg = {"warmup_candles": {"1m": 1000}}
    caplog.set_level(logging.WARNING)
    enabled = registry.filter_by_warmup(cfg, [stub])
    assert enabled == []
    assert "Insufficient warmup_candles" in caplog.text
