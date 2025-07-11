import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

# Load position_logger without importing crypto_bot.utils package
pkg = types.ModuleType("crypto_bot")
utils_pkg = types.ModuleType("crypto_bot.utils")
utils_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "crypto_bot/utils")]
sys.modules.setdefault("crypto_bot", pkg)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.position_logger",
    Path(__file__).resolve().parents[1] / "crypto_bot/utils/position_logger.py",
)
pl = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.utils.position_logger"] = pl
spec.loader.exec_module(pl)
spec_pw = importlib.util.spec_from_file_location(
    "crypto_bot.paper_wallet",
    Path(__file__).resolve().parents[1] / "crypto_bot/paper_wallet.py",
)
pw = importlib.util.module_from_spec(spec_pw)
sys.modules["crypto_bot.paper_wallet"] = pw
spec_pw.loader.exec_module(pw)
from crypto_bot.utils.logger import setup_logger
from crypto_bot.paper_wallet import PaperWallet


def test_log_position_writes_line(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    pl.log_position("XBT/USDT", "buy", 1.0, 100.0, 110.0, 1110.0)

    assert log_file.exists()
    text = log_file.read_text()
    assert "XBT/USDT" in text
    assert "$10.00" in text  # pnl in USD
    assert "positive" in text
    assert "100.000000" in text
    assert "110.000000" in text


def test_log_balance_writes_line(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test_balance", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    pl.log_balance(123.45)
    pl.log_balance(234.56)

    assert log_file.exists()
    text = log_file.read_text()
    assert "$123.45" in text


def test_close_trade_logs_realized_pnl(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test_close", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)
    pl.log_position("XBT/USDT", "buy", 1.0, 100.0, 100.0, wallet.balance)

    wallet.close("XBT/USDT", 1.0, 90.0)
    pl.log_position("XBT/USDT", "buy", 1.0, 100.0, 90.0, wallet.balance)

    assert log_file.exists()
    text = log_file.read_text()
    assert "$-10.00" in text
    assert "negative" in text
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1


def test_log_position_with_custom_pnl(tmp_path, monkeypatch):
    log_file = tmp_path / "positions.log"
    logger = setup_logger("pos_test_custom", str(log_file))
    monkeypatch.setattr(pl, "logger", logger)

    pl.log_position(
        "XBT/USDT",
        "buy",
        1.0,
        100.0,
        105.0,
        1105.0,
        pnl=15.0,
    )

    text = log_file.read_text()
    lines = log_file.read_text().splitlines()
    assert "$15.00" in text
    assert len(lines) == 2
    assert "pnl $0.00" in lines[0]
    assert "$-10.00" in lines[1]
