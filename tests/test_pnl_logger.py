import pandas as pd
from crypto_bot.utils import pnl_logger


def test_log_pnl_creates_csv(tmp_path, monkeypatch):
    log_file = tmp_path / "pnl.csv"
    monkeypatch.setattr(pnl_logger, "LOG_FILE", log_file)

    pnl_logger.log_pnl("trend_bot", "XBT/USDT", 100.0, 110.0, 10.0, 0.8, "buy")

    assert log_file.exists()
    df = pd.read_csv(log_file)
    expected_cols = {
        "timestamp",
        "strategy",
        "symbol",
        "entry_price",
        "exit_price",
        "pnl",
        "confidence",
        "direction",
    }
    assert expected_cols.issubset(df.columns)
