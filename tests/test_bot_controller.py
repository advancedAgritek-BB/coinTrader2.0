import pytest

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.bot_controller import TradingBotController
from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils import position_logger

@pytest.mark.asyncio
async def test_close_position_updates_wallet(monkeypatch, tmp_path):
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

    log_file = tmp_path / "positions.log"
    logger = setup_logger("ctl_balance", str(log_file))
    monkeypatch.setattr(position_logger, "logger", logger)

    result = await controller.close_position("XBT/USDT", 1.0)

    assert wallet.balance == pytest.approx(1010.0)
    assert "XBT/USDT" not in wallet.positions
    assert result["balance"] == pytest.approx(wallet.balance)

    assert log_file.exists()
    assert "$1010.00" in log_file.read_text()
