import json
from datetime import datetime, timedelta
import pytest
from crypto_bot import meta_selector
from crypto_bot.strategy import trend_bot, micro_scalp_bot
from crypto_bot.strategy_router import strategy_for


def test_choose_best_selects_highest(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    ts = datetime.utcnow().isoformat()
    data = {
        "trending": {
            "trend_bot": [
                {"pnl": 1.0, "timestamp": ts},
                {"pnl": -0.5, "timestamp": ts},
                {"pnl": 2.0, "timestamp": ts},
            ],
            "grid_bot": [
                {"pnl": 0.5, "timestamp": ts},
                {"pnl": -0.2, "timestamp": ts},
            ],
        }
    }
    file.write_text(json.dumps(data))
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    fn = meta_selector.choose_best("trending")
    assert fn is trend_bot.generate_signal


def test_choose_best_fallback(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    fn = meta_selector.choose_best("sideways")
    assert fn is strategy_for("sideways")


def test_strategy_map_contains_micro_scalp():
    assert (
        meta_selector._STRATEGY_FN_MAP.get("micro_scalp")
        is micro_scalp_bot.generate_signal
    )


def test_strategy_map_contains_dca_bot():
    from crypto_bot.strategy import dca_bot

    assert (
        meta_selector._STRATEGY_FN_MAP.get("dca_bot")
        is dca_bot.generate_signal
    )


def test_strategy_map_contains_bounce_scalper():
    from crypto_bot.strategy import bounce_scalper

    assert (
        meta_selector._STRATEGY_FN_MAP.get("bounce_scalper")
        is bounce_scalper.generate_signal
    )


def test_strategy_map_contains_solana_scalping():
    from crypto_bot.strategy import solana_scalping

    assert (
        meta_selector._STRATEGY_FN_MAP.get("solana_scalping")
        is solana_scalping.generate_signal
    )


def test_get_strategy_by_name_returns_callable():
    for name, fn in meta_selector._STRATEGY_FN_MAP.items():
        returned = meta_selector.get_strategy_by_name(name)
        assert returned is fn
        assert callable(returned)


def test_scores_penalize_drawdown(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    ts = datetime.utcnow().isoformat()
    data = {
        "trending": {
            "trend_bot": [
                {"pnl": 1.0, "timestamp": ts},
                {"pnl": 0.5, "timestamp": ts},
            ],
        }
    }
    file.write_text(json.dumps(data))
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    scores = meta_selector._scores_for("trending")
    assert scores["trend_bot"] == pytest.approx(1.75)


def test_scores_floor_at_zero(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    ts = datetime.utcnow().isoformat()
    data = {
        "trending": {
            "trend_bot": [
                {"pnl": 1.0, "timestamp": ts},
                {"pnl": -5.0, "timestamp": ts},
            ],
        }
    }
    file.write_text(json.dumps(data))
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    scores = meta_selector._scores_for("trending")
    assert scores["trend_bot"] == 0.0


def test_choose_best_uses_meta_regressor(tmp_path, monkeypatch):
    ts = datetime.utcnow().isoformat()
    data = {
        "trending": {
            "trend_bot": [{"pnl": 1.0, "timestamp": ts}],
            "grid_bot": [{"pnl": 0.1, "timestamp": ts}],
        }
    }
    perf = tmp_path / "perf.json"
    perf.write_text(json.dumps(data))
    monkeypatch.setattr(meta_selector, "LOG_FILE", perf)

    model_file = tmp_path / "m.txt"
    model_file.write_text("0")
    monkeypatch.setattr(meta_selector.MetaRegressor, "MODEL_PATH", model_file)

    def fake_predict(cls, regime, stats):
        return {"trend_bot": 0.1, "grid_bot": 0.3}

    monkeypatch.setattr(meta_selector.MetaRegressor, "predict_scores", classmethod(fake_predict))

    fn = meta_selector.choose_best("trending")
    assert fn is meta_selector.grid_bot.generate_signal


def test_recency_weighting():
    now = datetime.utcnow()
    trades = [
        {"pnl": 1.0, "timestamp": (now - timedelta(days=5)).isoformat()},
        {"pnl": 1.0, "timestamp": now.isoformat()},
    ]

    stats = meta_selector._compute_stats(trades)

    expected_old = 1.0 * (0.98 ** 5)
    expected_new = 1.0
    series = pytest.importorskip("pandas").Series([expected_old, expected_new])
    expected_sharpe = series.mean() / series.std() * (len(series) ** 0.5)

    assert stats["raw_sharpe"] == pytest.approx(expected_sharpe)
    assert expected_old / expected_new == pytest.approx(0.98 ** 5)


def test_min_sample_returns_empty_scores(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    data = {"trending": {"trend_bot": []}}
    file.write_text(json.dumps(data))
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    assert meta_selector._scores_for("trending") == {}


def test_risk_adjustment_formula(tmp_path, monkeypatch):
    ts = datetime.utcnow().isoformat()
    trades = [
        {"pnl": 1.0, "timestamp": ts},
        {"pnl": -0.5, "timestamp": ts},
        {"pnl": 2.0, "timestamp": ts},
    ]
    file = tmp_path / "perf.json"
    file.write_text(json.dumps({"trending": {"sample": trades}}))
    monkeypatch.setattr(meta_selector, "LOG_FILE", file)

    scores = meta_selector._scores_for("trending")
    stats = meta_selector._stats_for("trending")["sample"]

    expected = (
        stats["win_rate"]
        * stats["raw_sharpe"]
        / (1 + stats["downside_std"] + stats["max_dd"])
    )
    expected -= 0.5 * stats["max_dd"]
    expected = max(expected, 0.0)

    assert scores["sample"] == pytest.approx(expected)

