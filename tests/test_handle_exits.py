import pytest
pytest.importorskip("pandas")
import asyncio
import pandas as pd

import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext


def test_handle_exits_converts_profit(monkeypatch):
    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={"timeframe": "1h", "exit_strategy": {}})
    ctx.positions["XBT/USDT"] = {
        "side": "buy",
        "entry_price": 100.0,
        "size": 1.0,
        "trailing_stop": 0.0,
    }
    ctx.df_cache["1h"] = {"XBT/USDT": pd.DataFrame({"close": [110]})}
    ctx.risk_manager = type("RM", (), {"deallocate_capital": lambda *a, **k: None})()
    ctx.config["execution_mode"] = "dry_run"
    ctx.config["use_websocket"] = False
    ctx.config["solana_slippage_bps"] = 50
    ctx.exchange = object()
    ctx.ws_client = None
    ctx.notifier = None
    ctx.paper_wallet = None
    ctx.user_wallet = "wallet"

    monkeypatch.setattr(main, "should_exit", lambda *a, **k: (True, 0.0))
    async def fake_trade(*a, **k):
        pass
    monkeypatch.setattr(main, "cex_trade_async", fake_trade)

    called = {}
    async def fake_convert(wallet, from_t, to_t, amt, **kwargs):
        called["args"] = (wallet, from_t, to_t, amt)
        return {}
    monkeypatch.setattr(main, "auto_convert_funds", fake_convert)
    monkeypatch.setattr(main, "log_position", lambda *a, **k: None)

    asyncio.run(main.handle_exits(ctx))

    assert called["args"] == ("wallet", "USDT", "BTC", 10.0)

