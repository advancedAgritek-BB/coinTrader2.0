import pandas as pd
from crypto_bot.utils import metrics_logger as ml


def test_log_metrics_header_once(tmp_path):
    file = tmp_path / "metrics.csv"
    ml.log_metrics_to_csv({"a": 1, "timestamp": "t1"}, file)
    ml.log_metrics_to_csv({"a": 2, "timestamp": "t2"}, file)
    lines = file.read_text().strip().splitlines()
    assert len(lines) == 3
    header = lines[0].split(',')
    assert "a" in header and "timestamp" in header


def test_log_metrics_appends(tmp_path):
    file = tmp_path / "metrics.csv"
    ml.log_metrics_to_csv({"val": 1, "timestamp": "t1"}, file)
    ml.log_metrics_to_csv({"val": 2, "timestamp": "t2"}, file)
    df = pd.read_csv(file)
    assert df["val"].tolist() == [1, 2]


def test_log_cycle_latency(tmp_path):
    file = tmp_path / "metrics.csv"
    ml.log_cycle(0.1, 0.2, 0.3, 0.6, 0.4, 0.5, file)
    df = pd.read_csv(file)
    assert "ohlcv_fetch_latency" in df.columns
    assert "execution_latency" in df.columns
    assert df["ohlcv_fetch_latency"].iloc[0] == 0.4
    assert df["execution_latency"].iloc[0] == 0.5
