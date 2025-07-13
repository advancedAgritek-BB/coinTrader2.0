import pandas as pd
from crypto_bot.utils.telemetry import Telemetry


def test_counters_increment_and_reset(tmp_path):
    t = Telemetry()
    t.inc("a")
    t.inc("a", 2)
    t.inc("b")
    assert t.snapshot() == {"a": 3, "b": 1}
    t.reset()
    assert t.snapshot() == {}


def test_export_csv(tmp_path):
    t = Telemetry()
    t.inc("x", 2)
    file = tmp_path / "telemetry.csv"
    t.export_csv(file)
    df = pd.read_csv(file)
    assert df["x"].iloc[0] == 2
    assert "timestamp" in df.columns
