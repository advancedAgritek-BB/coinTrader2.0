import asyncio
from pathlib import Path

from crypto_bot.utils import trade_logger
from crypto_bot.execution import cex_executor
from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils.telegram import TelegramNotifier


def test_log_trade_appends_row(tmp_path, monkeypatch):
    log_file = tmp_path / "trades.csv"
    orig_path = trade_logger.Path
    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})

    def fake_path(p):
        if p == "crypto_bot/logs/trades.csv":
            return log_file
        return orig_path(p)

    monkeypatch.setattr(trade_logger, "Path", fake_path)

    order = {"symbol": "BTC/USDT", "side": "buy", "amount": 1}
    trade_logger.log_trade(order)
    trade_logger.log_trade(order)

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2


def test_execute_trade_async_logs(tmp_path, monkeypatch):
    trades = tmp_path / "trades.csv"
    exec_log = tmp_path / "execution.log"

    orig_path = trade_logger.Path
    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})

    def fake_path(p):
        if p == "crypto_bot/logs/trades.csv":
            return trades
        return orig_path(p)

    monkeypatch.setattr(trade_logger, "Path", fake_path)

    logger = setup_logger("exec_test", str(exec_log))
    monkeypatch.setattr(cex_executor, "logger", logger)

    def fake_send(self, text):
        logger.info(text)

    monkeypatch.setattr(TelegramNotifier, "notify", fake_send)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            object(),
            None,
            "BTC/USDT",
            "buy",
            1.0,
            TelegramNotifier("t", "c"),
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

    orig_path = trade_logger.Path
    monkeypatch.setattr(trade_logger, "dotenv_values", lambda _: {})

    def fake_path(p):
        if p == "crypto_bot/logs/trades.csv":
            return trades
        if p == "crypto_bot/logs/execution.log":
            return exec_log
        return orig_path(p)

    monkeypatch.setattr(trade_logger, "Path", fake_path)

    logger = setup_logger("stop_test", str(exec_log))
    monkeypatch.setattr(trade_logger, "logger", logger)

    order = {"symbol": "BTC/USDT", "side": "sell", "amount": 1, "stop": 9000}
    trade_logger.log_trade(order, is_stop=True)

    rows = trades.read_text().strip().splitlines()
    assert len(rows) == 1
    fields = rows[0].split(",")
    assert fields[5] == "True"
    assert fields[6] == "9000"
    assert "Stop order placed" in exec_log.read_text()
