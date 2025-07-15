import pytest
import crypto_bot.strategy_router as sr
from crypto_bot.strategy_router import RouterConfig, BotStats


def make_bot(name):
    def bot(df, cfg=None):
        return 0.0, "long"
    bot.__name__ = name
    return bot


def test_higher_score_selected(monkeypatch):
    bot_a = make_bot("bot_a")
    bot_b = make_bot("bot_b")
    monkeypatch.setattr(sr, "get_strategy_by_name", lambda n: {"bot_a": bot_a, "bot_b": bot_b}.get(n))
    def fake_stats(name):
        if name == "bot_a":
            return BotStats(sharpe_30d=1.0, win_rate_30d=0.5, avg_r_multiple=0.5)
        return BotStats()
    monkeypatch.setattr(sr, "load_bot_stats", fake_stats)
    cfg = RouterConfig.from_dict({"strategy_router": {"regimes": {"trending": ["bot_b", "bot_a"]}}})
    result = sr.get_strategies_for_regime("trending", cfg)
    assert result[0] is bot_a
    assert result[1] is bot_b


def test_equal_scores_use_order(monkeypatch):
    bot_a = make_bot("bot_a")
    bot_b = make_bot("bot_b")
    monkeypatch.setattr(sr, "get_strategy_by_name", lambda n: {"bot_a": bot_a, "bot_b": bot_b}.get(n))
    monkeypatch.setattr(sr, "load_bot_stats", lambda n: BotStats())
    cfg = RouterConfig.from_dict({"strategy_router": {"regimes": {"trending": ["bot_a", "bot_b"]}}})
    result = sr.get_strategies_for_regime("trending", cfg)
    assert result == [bot_a, bot_b]

