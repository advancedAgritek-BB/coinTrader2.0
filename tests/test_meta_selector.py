import json
from crypto_bot import meta_selector
from crypto_bot.strategy import trend_bot, micro_scalp_bot
from crypto_bot.strategy_router import strategy_for


def test_choose_best_selects_highest(tmp_path, monkeypatch):
    file = tmp_path / "perf.json"
    data = {
        "trending": {
            "trend_bot": [{"pnl": 1.0}, {"pnl": -0.5}, {"pnl": 2.0}],
            "grid_bot": [{"pnl": 0.5}, {"pnl": -0.2}],
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
