import pandas as pd
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig


def test_compute_strategy_weights_normalizes(tmp_path):
    file = tmp_path / "pnl.csv"
    data = [
        {"strategy": "trend", "pnl": 1},
        {"strategy": "trend", "pnl": -0.5},
        {"strategy": "grid", "pnl": 2},
        {"strategy": "grid", "pnl": 2},
    ]
    pd.DataFrame(data).to_csv(file, index=False)
    weights = compute_strategy_weights(file)
    assert set(weights.keys()) == {"trend", "grid"}
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6
    assert weights["grid"] > weights["trend"]


def test_risk_manager_updates_tracker():
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    rm = RiskManager(cfg)
    rm.update_allocation({"trend": 0.6, "grid": 0.4})
    assert rm.capital_tracker.allocation == {"trend": 0.6, "grid": 0.4}
