import types
import sys

import pandas as pd
import ta

# Provide a lightweight stub for pattern_detector to avoid heavy dependencies
stub = types.ModuleType("crypto_bot.regime.pattern_detector")
stub.detect_patterns = lambda *_a, **_k: {}
sys.modules["crypto_bot.regime.pattern_detector"] = stub

from crypto_bot.regime.regime_classifier import (
    classify_regime,
    clear_indicator_cache,
)


def _sample_df(rows: int = 30) -> pd.DataFrame:
    data = {
        "open": list(range(rows)),
        "high": [i + 1 for i in range(rows)],
        "low": list(range(rows)),
        "close": list(range(rows)),
        "volume": [100] * rows,
        "timestamp": list(range(rows)),
    }
    return pd.DataFrame(data)


def test_indicator_computations_are_cached(monkeypatch):
    df = _sample_df()
    key = ("BTC/USD", "1h")

    calls = {"ema": 0, "adx": 0}
    orig_ema = ta.trend.ema_indicator
    orig_adx = ta.trend.adx

    def counting_ema(series, window):
        calls["ema"] += 1
        return orig_ema(series, window)

    def counting_adx(high, low, close, window):
        calls["adx"] += 1
        return orig_adx(high, low, close, window)

    monkeypatch.setattr(ta.trend, "ema_indicator", counting_ema)
    monkeypatch.setattr(ta.trend, "adx", counting_adx)

    clear_indicator_cache(*key)
    classify_regime(df, symbol=key[0], cache_key=key)
    first_ema = calls["ema"]
    first_adx = calls["adx"]

    classify_regime(df, symbol=key[0], cache_key=key)
    assert calls["ema"] == first_ema
    assert calls["adx"] == first_adx

    df2 = df.copy()
    last = df2.iloc[-1]
    df2.loc[len(df2)] = {
        "open": last["open"] + 1,
        "high": last["high"] + 1,
        "low": last["low"] + 1,
        "close": last["close"] + 1,
        "volume": 100,
        "timestamp": int(last["timestamp"]) + 1,
    }
    classify_regime(df2, symbol=key[0], cache_key=key)
    assert calls["ema"] > first_ema
    assert calls["adx"] > first_adx

    clear_indicator_cache(*key)
