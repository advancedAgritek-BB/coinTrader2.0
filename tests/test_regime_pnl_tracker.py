import pandas as pd
import json
import pytest
from crypto_bot.utils import regime_pnl_tracker as rpt


def test_metrics_accumulate(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

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
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    rpt.log_trade("breakout", "dex_scalper", 1.0)
    rpt.log_trade("breakout", "dex_scalper", 1.5)
    rpt.log_trade("breakout", "grid_bot", 0.5)

    weights = rpt.compute_weights("breakout", log)
    assert set(weights.keys()) == {"dex_scalper", "grid_bot"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["dex_scalper"] > weights["grid_bot"]


def test_recent_win_rate(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    for pnl in [1.0, -1.0, 0.5]:
        rpt.log_trade("trending", "trend_bot", pnl)

    rate = rpt.get_recent_win_rate(2, log)
    assert rate == pytest.approx(0.5346, rel=1e-3)


def test_recent_win_rate_filters_by_strategy(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    # two strategies interleaved
    rpt.log_trade("trending", "trend_bot", 1.0)
    rpt.log_trade("trending", "grid_bot", -1.0)
    rpt.log_trade("trending", "trend_bot", 0.5)
    rpt.log_trade("trending", "grid_bot", 0.2)

    rate = rpt.get_recent_win_rate(3, log, strategy="trend_bot")
    assert rate == 1.0


def test_recent_win_rate_filters_strategy(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    rpt.log_trade("trending", "strat_a", 1.0)
    rpt.log_trade("trending", "strat_b", -1.0)
    rpt.log_trade("trending", "strat_a", 1.0)

    rate = rpt.get_recent_win_rate(5, log, strategy="strat_a")
    assert rate == 1.0


def test_win_rate_default(tmp_path, monkeypatch):
    """Empty logs should yield a bootstrap win rate of 0.6."""
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)
    assert rpt.get_recent_win_rate(5, log) == 0.6


def test_recent_win_rate_decay_weights_newer_trades(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    # Older losing trades followed by newer winners
    for pnl in [-1.0, -1.0, 1.0, 1.0, 1.0]:
        rpt.log_trade("trending", "trend_bot", pnl)

    # Strong decay so older losses have little influence
    rate = rpt.get_recent_win_rate(5, log, strategy="trend_bot", half_life=1)
    assert rate > 0.9


def test_log_trade_writes_performance_json(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    perf = tmp_path / "perf.json"
    monkeypatch.setattr(rpt, "LOG_FILE", log)
    monkeypatch.setattr(rpt, "PERF_FILE", perf)

    rpt.log_trade("trending", "trend_bot", 0.5)

    data = json.loads(perf.read_text())
    assert data["trending"]["trend_bot"][0]["pnl"] == 0.5
