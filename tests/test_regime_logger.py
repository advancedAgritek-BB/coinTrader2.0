import pandas as pd
from crypto_bot.utils import regime_logger as rl


def test_log_regime_appends_row(tmp_path, monkeypatch):
    log_file = tmp_path / "regime.csv"
    monkeypatch.setattr(rl, "LOG_FILE", log_file)
    rl.log_regime("XBT/USDT", "trending", 1.5)
    rl.log_regime("ETH/USDT", "sideways", -0.5)
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 2
    row1 = lines[0].split(',')
    assert row1[0] == "XBT/USDT"
    assert row1[1] == "trending"
    assert float(row1[2]) == 1.5


def test_summarize_accuracy(tmp_path, monkeypatch):
    log_file = tmp_path / "regime.csv"
    monkeypatch.setattr(rl, "LOG_FILE", log_file)
    data = [
        ["XBT/USDT", "trending", 1.0],
        ["ETH/USDT", "trending", 2.0],
        ["ADA/USDT", "sideways", -1.0],
    ]
    pd.DataFrame(data).to_csv(log_file, header=False, index=False)
    summary = rl.summarize_accuracy(log_file)
    assert summary["trending"] == 1.5
    assert summary["sideways"] == -1.0
