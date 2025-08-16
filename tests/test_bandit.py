import json
from crypto_bot.selector.bandit import Bandit
from crypto_bot.strategy_router import route
from crypto_bot.strategy import grid_bot, trend_bot


def test_bandit_update_and_state(tmp_path):
    state = tmp_path / "bandit.json"
    b = Bandit(state_file=str(state))
    wins = 0
    for i in range(50):
        win = i % 2 == 0
        if win:
            wins += 1
        b.update("BTC", "trend_bot", win)
    assert b.update_count == 50
    assert state.exists()
    data = json.loads(state.read_text())
    assert data["count"] == 50
    stats = data["priors"]["BTC"]["trend_bot"]
    assert stats["trades"] == 50
    assert stats["alpha"] == 1.0 + wins
    assert stats["beta"] == 1.0 + (50 - wins)


def test_route_uses_bandit(monkeypatch):
    calls = {}

    def fake_select(ctx, arms, symbol):
        calls["arms"] = list(arms)
        return "grid_bot"

    from crypto_bot import selector

    monkeypatch.setattr(selector.bandit, "enabled", True, raising=False)
    monkeypatch.setattr(selector.bandit, "select", fake_select)

    cfg = {
        "timeframe": "1h",
        "strategy_router": {"regimes": {"trending": ["trend_bot", "grid_bot"]}},
        "bandit": {"enabled": True},
    }
    fn = route("trending", "cex", cfg)

    assert calls.get("arms") == ["trend_bot", "grid_bot"]
    assert fn.__name__ == grid_bot.generate_signal.__name__


def test_route_filters_missing_strategies(monkeypatch):
    calls = {}

    def fake_select(ctx, arms, symbol):
        calls["arms"] = list(arms)
        return arms[0]

    from crypto_bot import selector

    monkeypatch.setattr(selector.bandit, "enabled", True, raising=False)
    monkeypatch.setattr(selector.bandit, "select", fake_select)

    cfg = {
        "timeframe": "1h",
        "strategy_router": {"regimes": {"trending": ["does_not_exist"]}},
        "bandit": {"enabled": True},
    }

    fn = route("trending", "cex", cfg)

    assert calls.get("arms") == ["generate_signal"]
    assert fn.__name__ == grid_bot.generate_signal.__name__
