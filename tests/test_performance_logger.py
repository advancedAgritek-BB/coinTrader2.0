import json
from crypto_bot.utils import performance_logger as pl


def test_log_performance_grouping(tmp_path, monkeypatch):
    log = tmp_path / "perf.json"
    monkeypatch.setattr(pl, "LOG_FILE", log)

    rec1 = {
        "symbol": "SOL/USDC",
        "regime": "trending",
        "strategy": "trend_bot",
        "pnl": 1.0,
        "entry_time": "e1",
        "exit_time": "x1",
    }
    rec2 = {
        "symbol": "SOL/USDC",
        "regime": "trending",
        "strategy": "trend_bot",
        "pnl": 2.0,
        "entry_time": "e2",
        "exit_time": "x2",
    }
    pl.log_performance(rec1)
    pl.log_performance(rec2)

    data = json.loads(log.read_text())
    assert "trending" in data
    assert "trend_bot" in data["trending"]
    assert len(data["trending"]["trend_bot"]) == 2
    assert data["trending"]["trend_bot"][0]["pnl"] == 1.0
    assert data["trending"]["trend_bot"][1]["pnl"] == 2.0
