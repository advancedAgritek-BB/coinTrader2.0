import json

import pandas as pd
import yaml

from crypto_bot import auto_optimizer
from crypto_bot.backtest.backtest_runner import BacktestRunner


def test_optimize_strategies_writes_best_params(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.02,
                "sharpe": 1.0,
                "max_drawdown": 0.1,
            },
            {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
                "sharpe": 0.5,
                "max_drawdown": 0.05,
            },
        ]
    )
    monkeypatch.setattr(BacktestRunner, "run_grid", lambda self: data)

    cfg = {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "mode": "cex",
        "risk": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        "optimization": {
            "enabled": True,
            "parameter_ranges": {
                "trend": {"stop_loss": [0.01, 0.02], "take_profit": [0.02, 0.03]}
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))
    monkeypatch.setattr(auto_optimizer, "CONFIG_PATH", config_path)
    out_file = tmp_path / "optimized.json"
    monkeypatch.setattr(auto_optimizer, "LOG_FILE", out_file)

    params = auto_optimizer.optimize_strategies()
    saved = json.loads(out_file.read_text())

    assert params == saved
    assert "trend" in saved
    assert saved["trend"]["stop_loss_pct"] == 0.01
    assert saved["trend"]["take_profit_pct"] == 0.02
