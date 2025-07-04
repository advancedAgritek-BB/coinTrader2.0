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
