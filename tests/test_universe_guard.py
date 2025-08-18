import asyncio
import logging
import types
import pytest

from crypto_bot.utils import symbol_utils as su

@pytest.fixture(autouse=True)
def reset_symbol_utils(monkeypatch):
    su._cached_symbols = None
    su._last_refresh = 0
    su._LAST_INVALIDATION_TS = 0
    monkeypatch.setattr(su, "_AUTO_FALLBACK_WARNED", False, raising=False)
    yield

def test_auto_mode_falls_back_to_cex(monkeypatch, caplog):
    async def fake_filter_symbols(exchange, symbols, config):
        return [("BTC/USDT", 1.0)], []
    monkeypatch.setattr(su, "filter_symbols", fake_filter_symbols, raising=False)
    cfg = {"mode": "auto", "symbol_filter": {}, "symbols": ["BTC/USDT"]}
    ex = types.SimpleNamespace(markets={})
    caplog.set_level(logging.DEBUG)
    scored, onchain = asyncio.run(su.get_filtered_symbols(ex, cfg))
    assert scored == [("BTC/USDT", 1.0)]
    assert onchain == []
    assert cfg["mode"] == "cex"
    assert "using CEX mode" in caplog.text

def test_empty_universe_raises(monkeypatch):
    async def fake_filter_symbols(exchange, symbols, config):
        return [("BTC/USDT", 1.0)], []
    monkeypatch.setattr(su, "filter_symbols", fake_filter_symbols, raising=False)
    cfg = {"mode": "onchain", "symbol_filter": {}, "symbols": ["BTC/USDT"]}
    ex = types.SimpleNamespace(markets={})
    with pytest.raises(RuntimeError, match=r"Universe is empty \(mode=onchain\)"):
        asyncio.run(su.get_filtered_symbols(ex, cfg))
