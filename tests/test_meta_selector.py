import json
from datetime import datetime
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


def test_strategy_map_contains_bounce_scalper():
    from crypto_bot.strategy import bounce_scalper

    assert (
        meta_selector._STRATEGY_FN_MAP.get("bounce_scalper")
        is bounce_scalper.generate_signal
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

