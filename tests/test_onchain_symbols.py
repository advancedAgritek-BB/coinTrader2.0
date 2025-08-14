import asyncio
import types
import sys
from collections import deque
import pandas as pd
import pytest

dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, dummy)
oauth_module = types.ModuleType("oauth2client")
oauth_module.service_account = types.SimpleNamespace(ServiceAccountCredentials=object)
sys.modules.setdefault("oauth2client", oauth_module)
sys.modules.setdefault("oauth2client.service_account", oauth_module.service_account)

import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext
from crypto_bot.utils import symbol_utils


class StopLoop(Exception):
    pass


def setup_main_common(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace())
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    async def fake_send_test(*_a, **_k):
        return True
    monkeypatch.setattr(main, "send_test_message", fake_send_test)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)
    monkeypatch.setattr(main.asyncio, "sleep", lambda *_a: None)
    monkeypatch.setattr(main, "MAX_SYMBOL_SCAN_ATTEMPTS", 1)
    monkeypatch.setattr(main, "SYMBOL_SCAN_RETRY_DELAY", 0)
    monkeypatch.setattr(main, "MAX_SYMBOL_SCAN_DELAY", 0)

    class DummyRC:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(main, "RiskConfig", DummyRC)

    class DummyExchange:
        def fetch_balance(self):
            return {"USDT": {"free": 0}}

    captured = {}

    def fake_get_exchange(conf):
        captured["cfg"] = conf
        return DummyExchange(), None

    monkeypatch.setattr(main, "get_exchange", fake_get_exchange)
    return captured


def test_scan_runs_with_onchain_only(monkeypatch):
    cfg = {"onchain_symbols": ["SOL/USDC"], "scan_markets": True}
    captured = setup_main_common(monkeypatch, cfg)

    calls = {"loader": 0}

    async def fake_loader(ex, exclude=None, config=None):
        calls["loader"] += 1
        return ["BTC/USD"]

    monkeypatch.setattr(main, "load_kraken_symbols", fake_loader)

    class DummyRM:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(main, "RiskManager", DummyRM)

    asyncio.run(main.main())

    assert calls["loader"] == 1
    assert captured["cfg"]["symbols"] == ["BTC/USD", "SOL/USDC"]


def test_get_filtered_symbols_onchain(monkeypatch, caplog):
    caplog.set_level("INFO")

    async def fake_filter_symbols(ex, syms, cfg):
        return [("ETH/USD", 1.0)]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)
    cfg = {"symbols": ["ETH/USD"], "onchain_symbols": ["BONK/USDC"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    res = asyncio.run(symbol_utils.get_filtered_symbols(object(), cfg))
    assert res[0] == [("ETH/USD", 1.0)]
    assert res[1] == ["BONK/USDC"]
    assert not any("Dropping invalid USDC pair" in r.getMessage() for r in caplog.records)


def test_fetch_candidates_adds_onchain(monkeypatch):
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={"timeframe": "1h", "symbol_batch_size": 2},
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], ["BONK/USDC"]

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "build_priority_queue", lambda pairs: deque([s for s, _ in pairs]))
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.01)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)

    asyncio.run(main.fetch_candidates(ctx))

    assert set(ctx.current_batch) == {"BTC/USD", "BONK/USDC"}


def test_auto_mode_falls_back_to_cex(monkeypatch, caplog):
    caplog.set_level("INFO")
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={"timeframe": "1h", "symbol_batch_size": 2, "mode": "auto"},
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "build_priority_queue", lambda pairs: deque([s for s, _ in pairs]))
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.01)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.resolved_mode == "cex"
    assert "falling back to CEX" in caplog.text
    assert "BTC/USD" in ctx.active_universe
