import pandas as pd
from crypto_bot.utils import rank_logger


def test_log_second_place_creates_file(tmp_path, monkeypatch):
    file = tmp_path / "second.csv"
    monkeypatch.setattr(rank_logger, "LOG_FILE", file)
    rank_logger.log_second_place("XBT/USDT", "trending", "trend_bot", 0.6, 0.1)
    rank_logger.log_second_place("ETH/USDT", "sideways", "grid_bot", 0.5, 0.05)
    assert file.exists()
    df = pd.read_csv(file)
    assert len(df) == 2
    assert set(df.columns) == {"timestamp", "symbol", "regime", "strategy", "score", "edge"}

