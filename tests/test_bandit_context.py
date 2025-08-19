import pandas as pd
import sys
import types

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
class _FakeTelegram:
    class Bot:
        def __init__(self, *a, **k):
            pass

sys.modules.setdefault("telegram", _FakeTelegram())
import crypto_bot.strategy_router as router


def _df():
    idx = pd.date_range("2020-01-01", periods=2, freq="1h")
    data = {
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [1.0, 1.0],
    }
    return pd.DataFrame(data, index=idx)


def test_bandit_context_uses_pyth(monkeypatch):
    df = _df()

    import crypto_bot.utils.pyth as pyth_mod
    monkeypatch.setattr(pyth_mod, "get_pyth_price", lambda symbol: 2.0)

    import crypto_bot.volatility_filter as vf

    monkeypatch.setattr(vf, "calc_atr", lambda d: pd.Series([1.0]))

    from crypto_bot.utils import stats as utils_stats

    monkeypatch.setattr(utils_stats, "zscore", lambda s, lookback=20: pd.Series([0, 0], index=s.index))

    ctx = router._bandit_context(df, "trending", "BTC/USD")
    assert abs(ctx.get("atr_pct") - 0.5) < 1e-6


def test_bandit_context_fallback(monkeypatch):
    df = _df()

    import crypto_bot.utils.pyth as pyth_mod
    monkeypatch.setattr(pyth_mod, "get_pyth_price", lambda symbol: None)
    import crypto_bot.volatility_filter as vf

    monkeypatch.setattr(vf, "calc_atr", lambda d: pd.Series([1.0]))

    from crypto_bot.utils import stats as utils_stats

    monkeypatch.setattr(utils_stats, "zscore", lambda s, lookback=20: pd.Series([0, 0], index=s.index))

    ctx = router._bandit_context(df, "trending", "BTC/USD")
    assert abs(ctx.get("atr_pct") - 1.0) < 1e-6
