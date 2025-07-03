from crypto_bot.utils import position_logger as pl
from crypto_bot.utils.logger import setup_logger


def test_log_position_writes_line(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    pl.log_position("BTC/USDT", "buy", 1.0, 100.0, 110.0, 1110.0)

    assert log_file.exists()
    text = log_file.read_text()
    assert "BTC/USDT" in text
    assert "positive" in text


def test_log_balance_writes_line(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test_balance", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    pl.log_balance(123.45)

    assert log_file.exists()
    text = log_file.read_text()
    assert "123.45" in text
