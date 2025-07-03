import asyncio
from pathlib import Path

from crypto_bot.utils import trade_logger
from crypto_bot.execution import cex_executor
from crypto_bot.utils.logger import setup_logger


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

    def fake_send(token, chat_id, text):
        logger.info(text)

    monkeypatch.setattr(cex_executor, "send_message", fake_send)

    order = asyncio.run(
        cex_executor.execute_trade_async(
            object(),
            None,
            "BTC/USDT",
            "buy",
            1.0,
            "t",
            "c",
            dry_run=True,
        )
    )

    assert order["dry_run"] is True
    assert trades.exists()
    assert exec_log.exists()
    assert "Placing buy order" in exec_log.read_text()
    assert len(trades.read_text().strip().splitlines()) == 1
