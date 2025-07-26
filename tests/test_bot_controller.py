import pytest

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.bot_controller import TradingBotController

@pytest.mark.asyncio
async def test_close_position_updates_wallet(monkeypatch):
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)

    controller = TradingBotController.__new__(TradingBotController)
    controller.exchange = object()
    controller.ws_client = None
    controller.config = {"execution_mode": "dry_run", "use_websocket": False}
    controller.paper_wallet = wallet

    async def fake_exec(*a, **k):
        return {"price": 110.0}

    monkeypatch.setattr("crypto_bot.bot_controller.execute_trade_async", fake_exec)

    await controller.close_position("XBT/USDT", 1.0)
    assert wallet.balance == pytest.approx(1010.0)
    assert "XBT/USDT" not in wallet.positions
