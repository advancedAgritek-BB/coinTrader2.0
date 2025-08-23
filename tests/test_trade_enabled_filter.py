import types

from crypto_bot.strategies import loader


def test_trade_enabled_attribute_filters_strategies(monkeypatch):
    """Strategies with ``trade_enabled`` set to ``False`` are skipped."""

    enabled_obj = types.SimpleNamespace(name="mean_revert", trade_enabled=True)
    disabled_obj = types.SimpleNamespace(name="trend_bot", trade_enabled=False)

    def fake_load_strategies(enabled=None):  # noqa: D401 - simple stub
        return {"mean_revert": enabled_obj, "trend_bot": disabled_obj}, {}

    monkeypatch.setattr(loader, "_load_strategies", fake_load_strategies)

    strategies = loader.load_strategies("cex")

    assert enabled_obj in strategies
    assert disabled_obj not in strategies

