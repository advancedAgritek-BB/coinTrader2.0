import asyncio
from types import SimpleNamespace

from crypto_bot.utils import symbol_utils as su
from crypto_bot.markets import symbol_service as ss


class DummyExchange:
    id = "dummy"

    def __init__(self):
        self._markets = {"AAA/USDT": {}, "BBB/USDT": {}}

    def list_markets(self, timeout=None):
        return list(self._markets.keys())

    @property
    def markets(self):
        return self._markets

    @markets.setter
    def markets(self, value):
        self._markets = value


async def _fake_filter_symbols(exchange, symbols, config):
    return [(s, 0.0) for s in symbols], []


def test_strict_cex_returns_only_listed(monkeypatch):
    cfg = SimpleNamespace(strict_cex=False, allowed_quotes=[], min_volume=0.0, denylist_symbols=[])
    monkeypatch.setattr(su, "cfg", cfg, raising=False)
    monkeypatch.setattr(ss, "cfg", cfg, raising=False)
    monkeypatch.setattr(su, "filter_symbols", _fake_filter_symbols)
    monkeypatch.setattr(ss.SymbolService, "_kraken_symbols", staticmethod(lambda: set()))
    su.invalidate_symbol_cache()

    config = {
        "mode": "cex",
        "strict_cex": True,
        "symbol_filter": {},
        "symbol_refresh_minutes": 0,
        "symbols": ["AAA/USDT", "CCC/USDT"],
    }
    exchange = DummyExchange()
    scored, onchain = asyncio.run(su.get_filtered_symbols(exchange, config))
    assert [s for s, _ in scored] == ["AAA/USDT"]
    assert onchain == []
