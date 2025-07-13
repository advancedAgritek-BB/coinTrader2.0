import pandas as pd
from crypto_bot.utils import regime_pnl_tracker as rpt


def test_metrics_accumulate(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    monkeypatch.setattr(rpt, "LOG_FILE", log)

    rpt.log_trade("trending", "trend_bot", 1.0)
    rpt.log_trade("trending", "trend_bot", -0.5)
    rpt.log_trade("trending", "trend_bot", 2.0)

    metrics = rpt.get_metrics("trending", log)
    stats = metrics["trending"]["trend_bot"]
    assert stats["pnl"] == 2.5
    assert round(stats["drawdown"], 6) == 0.5
    assert round(stats["sharpe"], 6) == round((1.0 - 0.5 + 2.0) / 3 / pd.Series([1.0, -0.5, 2.0]).std() * (3 ** 0.5), 6)


def test_compute_weights_normalizes(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    monkeypatch.setattr(rpt, "LOG_FILE", log)

    rpt.log_trade("breakout", "scalper", 1.0)
    rpt.log_trade("breakout", "scalper", 1.5)
    rpt.log_trade("breakout", "grid", 0.5)

    weights = rpt.compute_weights("breakout", log)
    assert set(weights.keys()) == {"scalper", "grid"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["scalper"] > weights["grid"]


def test_recent_win_rate(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    monkeypatch.setattr(rpt, "LOG_FILE", log)

    for pnl in [1.0, -1.0, 0.5]:
        rpt.log_trade("trending", "trend_bot", pnl)

    rate = rpt.get_recent_win_rate(2, log)
    assert rate == 0.5


def test_recent_win_rate_filters_by_strategy(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    monkeypatch.setattr(rpt, "LOG_FILE", log)

    # two strategies interleaved
    rpt.log_trade("trending", "trend_bot", 1.0)
    rpt.log_trade("trending", "grid_bot", -1.0)
    rpt.log_trade("trending", "trend_bot", 0.5)
    rpt.log_trade("trending", "grid_bot", 0.2)

    rate = rpt.get_recent_win_rate(3, log, strategy="trend_bot")
    assert rate == 1.0


def test_recent_win_rate_filters_strategy(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    monkeypatch.setattr(rpt, "LOG_FILE", log)

    rpt.log_trade("trending", "strat_a", 1.0)
    rpt.log_trade("trending", "strat_b", -1.0)
    rpt.log_trade("trending", "strat_a", 1.0)

    rate = rpt.get_recent_win_rate(5, log, strategy="strat_a")
    assert rate == 1.0
