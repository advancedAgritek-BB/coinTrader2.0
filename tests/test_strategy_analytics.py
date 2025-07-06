import json
import pytest
from crypto_bot.utils import strategy_analytics as sa


def test_compute_metrics(tmp_path):
    data = {
        "trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}, {"pnl": 2.0}],
        "grid_bot": [{"pnl": 0.1}, {"pnl": -0.2}],
    }
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(data))

    metrics = sa.compute_metrics(f)
    assert set(metrics.keys()) == {"trend_bot", "grid_bot"}
    trend = metrics["trend_bot"]
    assert pytest.approx(trend["win_rate"], rel=1e-2) == 2 / 3
    assert pytest.approx(trend["ev"], rel=1e-2) == (1 - 0.5 + 2) / 3
    assert pytest.approx(trend["drawdown"], rel=1e-2) == -0.5
    assert trend["sharpe"] > 0


def test_compute_metrics_nested(tmp_path):
    data = {
        "trending": {
            "trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}],
            "grid_bot": [{"pnl": 0.2}],
        },
        "sideways": {
            "mean_bot": [{"pnl": -1.0}, {"pnl": 0.5}],
        },
    }
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(data))

    metrics = sa.compute_metrics(f)
    assert set(metrics) == {"trend_bot", "grid_bot", "mean_bot"}


def test_compute_metrics_invalid(tmp_path):
    data = {"trend_bot": [{"pnl": 1.0}, 2.0]}
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        sa.compute_metrics(f)


def test_compute_strategy_stats(tmp_path):
    data = {
        "trend_bot": [{"pnl": 1.0}, {"pnl": -1.0}, {"pnl": 2.0}],
        "grid_bot": [{"pnl": 0.5}],
    }
    f = tmp_path / "perf.json"
    f.write_text(json.dumps(data))

    stats = sa.compute_strategy_stats(f)
    assert stats["trend_bot"]["trades"] == 3
    assert pytest.approx(stats["trend_bot"]["win_rate"], rel=1e-2) == 2 / 3
    assert pytest.approx(stats["trend_bot"]["avg_win"], rel=1e-2) == 1.5
    assert pytest.approx(stats["trend_bot"]["avg_loss"], rel=1e-2) == -1.0
    assert stats["grid_bot"]["trades"] == 1


def test_write_stats(tmp_path):
    data = {"trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}]}
    perf = tmp_path / "perf.json"
    perf.write_text(json.dumps(data))
    out = tmp_path / "stats.json"

    res = sa.write_stats(out, perf)
    loaded = json.loads(out.read_text())
    assert loaded == res
    assert "trend_bot" in res


