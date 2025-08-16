import asyncio
import pandas as pd
import crypto_bot.utils.market_analyzer as ma
from crypto_bot.utils.strategy_utils import compute_strategy_weights, compute_drawdown
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig


def test_compute_strategy_weights_normalizes(tmp_path):
    file = tmp_path / "pnl.csv"
    data = [
        {"strategy": "trend_bot", "pnl": 1},
        {"strategy": "trend_bot", "pnl": -0.5},
        {"strategy": "grid_bot", "pnl": 2},
        {"strategy": "grid_bot", "pnl": 2},
    ]
    pd.DataFrame(data).to_csv(file, index=False)
    weights = compute_strategy_weights(file, scoring_method="sharpe")
    assert set(weights.keys()) == {"trend_bot", "grid_bot"}
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6
    assert weights["trend_bot"] > weights["grid_bot"]


def test_risk_manager_updates_tracker():
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    rm = RiskManager(cfg)
    rm.update_allocation({"trend_bot": 0.6, "grid_bot": 0.4})
    assert rm.capital_tracker.allocation == {"trend_bot": 0.6, "grid_bot": 0.4}


def test_compute_drawdown_basic():
    df = pd.DataFrame({"close": [10, 12, 11, 15, 14]})
    dd = compute_drawdown(df, lookback=5)
    assert dd == -1


def test_run_candidates_weighting(monkeypatch):
    df = pd.DataFrame({"close": [1, 2]})

    def a(df, cfg=None):
        return 0.5, "long"

    def b(df, cfg=None):
        return 0.5, "long"

    async def fake_eval(strats, df_, cfg_, max_parallel=4):
        return [(0.5, "long", None) for _ in strats]

    monkeypatch.setattr(ma, "evaluate_async", fake_eval)
    monkeypatch.setattr(ma.perf, "edge", lambda name, sym, coef=0.0: 1.0)
    monkeypatch.setattr(ma, "compute_strategy_weights", lambda: {"a": 0.1, "b": 0.9})

    res = asyncio.run(ma.run_candidates(df, [a, b], "AAA", {}, None))
    names = [fn.__name__ for fn, _s, _d in res]
    assert names == ["b", "a"]
