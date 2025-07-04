import pandas as pd
from crypto_bot.rl import strategy_selector
from crypto_bot.strategy import trend_bot, grid_bot


def test_train_and_select(tmp_path):
    log = tmp_path / "pnl.csv"
    data = pd.DataFrame({
        "regime": ["trending", "trending", "sideways", "sideways"],
        "strategy": ["trend_bot", "grid_bot", "trend_bot", "grid_bot"],
        "pnl": [1.0, -0.5, -1.0, 2.0],
    })
    data.to_csv(log, index=False)

    selector = strategy_selector.RLStrategySelector()
    selector.train(log)

    assert selector.select("trending").__name__ == trend_bot.generate_signal.__name__
    assert selector.select("sideways").__name__ == grid_bot.generate_signal.__name__


def test_select_strategy_global(tmp_path, monkeypatch):
    log = tmp_path / "pnl.csv"
    df = pd.DataFrame({
        "regime": ["trending"],
        "strategy": ["grid_bot"],
        "pnl": [0.5],
    })
    df.to_csv(log, index=False)
    monkeypatch.setattr(strategy_selector, "LOG_FILE", log)

    # Reset global selector
    strategy_selector._selector = strategy_selector.RLStrategySelector()

    fn = strategy_selector.select_strategy("trending")
    assert fn.__name__ == grid_bot.generate_signal.__name__
