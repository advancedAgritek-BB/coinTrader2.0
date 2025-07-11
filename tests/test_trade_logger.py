import asyncio
from pathlib import Path

from crypto_bot.utils import trade_logger
from crypto_bot.execution import cex_executor


class DummyNotifier:
    def __init__(self, logger):
        self.logger = logger
        self.messages = []

    def notify(self, text: str):
        self.messages.append(text)
        self.logger.info(text)
        return None
from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils.telegram import TelegramNotifier


def test_log_trade_appends_row(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"
    exec_log = tmp_path / "execution.log"
    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})
    monkeypatch.setattr(trade_logger, "LOG_DIR", tmp_path)

    logger = setup_logger("trade_test", str(exec_log))
    monkeypatch.setattr(trade_logger, "logger", logger)

    order = {"symbol": "XBT/USDT", "side": "buy", "amount": 1}
    trade_logger.log_trade(order)

    rows = trades.read_text().strip().splitlines()
    assert len(rows) == 1
    assert exec_log.read_text().count("Logged trade") == 1


def test_execute_trade_async_logs(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"
    exec_log = tmp_path / "execution.log"

    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})
    monkeypatch.setattr(trade_logger, "LOG_DIR", tmp_path)

    logger = setup_logger("exec_test", str(exec_log))
    monkeypatch.setattr(cex_executor, "logger", logger)
    notifier = DummyNotifier(logger)

    def fake_send(self, text):
        logger.info(text)

    monkeypatch.setattr(TelegramNotifier, "notify", fake_send)
    monkeypatch.setattr(cex_executor.Notifier, "notify", fake_send)
    monkeypatch.setattr(cex_executor, "send_message", lambda *a, **k: logger.info(a[2]))
    monkeypatch.setattr(cex_executor.TelegramNotifier, "notify", fake_send)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            object(),
            None,
            "XBT/USDT",
            "buy",
            1.0,
            TelegramNotifier("t", "c"),
            notifier=notifier,
            dry_run=True,
        )
    )

    assert order["dry_run"] is True
    assert trades.exists()
    assert exec_log.exists()
    assert "Placing buy order" in exec_log.read_text()
    assert len(trades.read_text().strip().splitlines()) == 1


def test_stop_order_logged(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"
    exec_log = tmp_path / "execution.log"

    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})
    monkeypatch.setattr(trade_logger, "LOG_DIR", tmp_path)

    logger = setup_logger("stop_test", str(exec_log))
    monkeypatch.setattr(trade_logger, "logger", logger)

    order = {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "stop": 9000}
    trade_logger.log_trade(order, is_stop=True)

    rows = trades.read_text().strip().splitlines()
    assert len(rows) == 1
    fields = rows[0].split(",")
    assert fields[5] == "True"
    assert fields[6] == "9000"
    text = exec_log.read_text()
    assert text.count("Stop order placed") == 1
