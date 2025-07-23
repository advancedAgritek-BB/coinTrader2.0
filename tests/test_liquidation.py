import asyncio
import crypto_bot.main as main
from crypto_bot.phase_runner import BotContext
from crypto_bot.utils.logger import setup_logger
import logging

class DummyCtx:
    pass

def test_liquidate_state_calls_force_exit(monkeypatch):
    state = {"liquidate_all": True}
    called = {}

    async def fake_force_exit_all(ctx):
        called["hit"] = True

    monkeypatch.setattr(main, "force_exit_all", fake_force_exit_all)

    async def run():
        if state.get("liquidate_all"):
            await main.force_exit_all(DummyCtx())
            state["liquidate_all"] = False

    asyncio.run(run())

    assert called.get("hit") is True
    assert state["liquidate_all"] is False


def test_force_exit_all_logs(tmp_path, monkeypatch):
    log_file = tmp_path / "bot.log"
    logger = setup_logger("force_exit_test", str(log_file), to_console=False)
    monkeypatch.setattr(main, "logger", logger)

    async def fake_trade_async(*a, **k):
        pass

    async def fake_refresh(ctx):
        return 0.0

    monkeypatch.setattr(main, "cex_trade_async", fake_trade_async)
    monkeypatch.setattr(main, "refresh_balance", fake_refresh)
    monkeypatch.setattr(main, "log_position", lambda *a, **k: None)

    class DummyRM:
        def deallocate_capital(self, *a, **k):
            pass

    ctx = BotContext(
        positions={"XBT/USDT": {"side": "buy", "entry_price": 100.0, "size": 1.0}},
        df_cache={},
        regime_cache={},
        config={"timeframe": "1h", "execution_mode": "live"},
        exchange=object(),
        ws_client=None,
        risk_manager=DummyRM(),
        notifier=None,
        paper_wallet=None,
        position_guard=None,
    )

    asyncio.run(main.force_exit_all(ctx))

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.flush()

    text = log_file.read_text()
    assert "Liquidating XBT/USDT" in text
    assert "Liquidated XBT/USDT" in text
