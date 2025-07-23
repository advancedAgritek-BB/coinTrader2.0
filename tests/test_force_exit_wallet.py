import asyncio
import pandas as pd
import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext
from crypto_bot.paper_wallet import PaperWallet

class DummyRM:
    def deallocate_capital(self, *a, **k):
        pass


def test_force_exit_all_handles_wallet(monkeypatch):
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"XBT/USDT": pd.DataFrame({"close": [110.0]})}},
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

    monkeypatch.setattr(main, "cex_trade_async", lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(main, "refresh_balance", lambda _ctx: asyncio.sleep(0))
    monkeypatch.setattr(main, "log_position", lambda *a, **k: None)

    asyncio.run(main.force_exit_all(ctx))
    assert ctx.balance == wallet.balance

    assert ctx.paper_wallet.positions == {}

