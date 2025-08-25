import types
import pytest

from crypto_bot.strategy_manager import evaluate_all
from crypto_bot.strategies import loader as strategies_loader


@pytest.mark.asyncio
async def test_evaluate_all_with_mock_strategy(monkeypatch):
    async def dummy_score(symbols=None, timeframes=None):
        return {(symbols[0], timeframes[0]): {"score": 1.0, "signal": "buy"}}

    mock_strategy = types.SimpleNamespace(score=dummy_score, __name__="mock_strategy")

    def fake_load_strategies(package_name="crypto_bot.strategy", enabled=None):
        return {"mock": mock_strategy}, {}

    monkeypatch.setattr(strategies_loader, "_load_strategies", fake_load_strategies)

    result = await evaluate_all(["SYM"], ["1m"])

    assert isinstance(result, list)
    assert len(result) == 1
