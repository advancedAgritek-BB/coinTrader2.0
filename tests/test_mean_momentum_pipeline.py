import asyncio
import pandas as pd

import crypto_bot.utils.market_analyzer as ma
import crypto_bot.strategy_router as sr
from crypto_bot.strategy import mean_bot, momentum_bot


def _df_low_bw_drop() -> pd.DataFrame:
    import numpy as np

    np.random.seed(0)
    base = list(100 + np.random.randn(60) * 10)
    for i in range(40, 56):
        base[i] = 100 + np.random.randn() * 2
    base = base[:55] + [base[54] - 2]
    data = {
        "open": base,
        "high": [p + 1 for p in base],
        "low": [p - 1 for p in base],
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def _momentum_df(length: int = 40) -> pd.DataFrame:
    prices = list(range(1, length + 1))
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [100] * length,
        }
    )


def test_mean_bot_enqueues_trade(monkeypatch):
    df = _df_low_bw_drop()
    cfg = {
        "strategy_router": {"regimes": {"sideways": ["mean_bot"]}},
        "mean_bot": {"enabled": True},
    }
    monkeypatch.setattr(ma.perf, "edge", lambda *_args, **_kw: 1.0)
    strategies = sr.get_strategies_for_regime("sideways", cfg)
    res = asyncio.run(ma.run_candidates(df, strategies, "AAA", cfg, "sideways"))
    assert any(fn is mean_bot.generate_signal and d != "none" for fn, _s, d in res)


def test_momentum_bot_enqueues_trade(monkeypatch):
    df = _momentum_df()
    cfg = {
        "strategy_router": {"regimes": {"trending": ["momentum_bot"]}},
        "momentum_bot": {"enabled": True},
    }
    monkeypatch.setattr(ma.perf, "edge", lambda *_args, **_kw: 1.0)
    strategies = sr.get_strategies_for_regime("trending", cfg)
    res = asyncio.run(ma.run_candidates(df, strategies, "AAA", cfg, "trending"))
    assert any(fn is momentum_bot.generate_signal and d != "none" for fn, _s, d in res)
