import asyncio
import logging

from crypto_bot.utils import symbol_utils

class DummyExchange:
    pass


def test_get_filtered_symbols_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    async def fake_to_thread(fn, *a):
        return fn(*a)

    monkeypatch.setattr(symbol_utils, "filter_symbols", lambda ex, syms: [])
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == ["BTC/USD"]
    assert any("falling back" in r.getMessage() for r in caplog.records)
