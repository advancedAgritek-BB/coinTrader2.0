import json
import pandas as pd
from crypto_bot.utils import pnl_logger
from crypto_bot.selector import bandit


def test_log_pnl_creates_csv_and_json(tmp_path, monkeypatch):
    log_file = tmp_path / "pnl.csv"
    perf_file = tmp_path / "perf.json"
    monkeypatch.setattr(pnl_logger, "LOG_FILE", log_file)
    monkeypatch.setattr(pnl_logger, "PERFORMANCE_FILE", perf_file)

    calls = {}

    def fake_update(symbol, strategy, win):
        calls["args"] = (symbol, strategy, win)

    monkeypatch.setattr(bandit, "update", fake_update)

    pnl_logger.log_pnl(
        "bull",
        "trend_bot",
        "XBT/USDT",
        100.0,
        110.0,
        10.0,
        0.8,
        "buy",
    )

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
    assert len(df) == 1
    assert calls["args"] == ("XBT/USDT", "trend_bot", True)

    assert perf_file.exists()
    data = json.loads(perf_file.read_text())
    assert len(data["bull"]["trend_bot"]) == 1
    assert data["bull"]["trend_bot"][0]["pnl"] == 10.0
