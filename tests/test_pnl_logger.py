import pandas as pd
from crypto_bot.utils import pnl_logger
from crypto_bot.selector import bandit


def test_log_pnl_creates_csv(tmp_path, monkeypatch):
    log_file = tmp_path / "pnl.csv"
    monkeypatch.setattr(pnl_logger, "LOG_FILE", log_file)

    calls = {}

    def fake_update(symbol, strategy, win):
        calls["args"] = (symbol, strategy, win)

    monkeypatch.setattr(bandit, "update", fake_update)

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
    assert calls["args"] == ("XBT/USDT", "trend_bot", True)
