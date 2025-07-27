import asyncio
import pandas as pd
import crypto_bot.main as main
from crypto_bot.execution import cex_executor
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet

class DummyRM:
    def deallocate_capital(self, *a, **k):
        pass


def test_force_exit_all_handles_wallet(monkeypatch):
    wallet = PaperWallet(1000.0)
    start_balance = wallet.balance
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)
    ctx = BotContext(
        positions={},
        df_cache={"1h": {}},
        regime_cache={},
        config={"execution_mode": "dry_run", "timeframe": "1h"},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=wallet,
        position_guard=None,
    )
    ctx.balance = wallet.balance

    async def fake_exec(*args, **kwargs):
        symbol = args[2] if len(args) >= 3 else kwargs.get("symbol")
        side = args[3] if len(args) >= 4 else kwargs.get("side")
        amount = args[4] if len(args) >= 5 else kwargs.get("amount")
        return {"symbol": symbol, "side": side, "amount": amount}

    monkeypatch.setattr(cex_executor, "execute_trade_async", fake_exec)
    monkeypatch.setattr(main, "cex_trade_async", fake_exec)
    monkeypatch.setattr(main, "refresh_balance", lambda _ctx: asyncio.sleep(0))
    monkeypatch.setattr(main, "log_position", lambda *a, **k: None)

    asyncio.run(main.force_exit_all(ctx))
    assert ctx.balance == start_balance

    assert ctx.paper_wallet.positions == {}

