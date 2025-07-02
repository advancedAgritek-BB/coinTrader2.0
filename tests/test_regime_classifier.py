import pandas as pd

from crypto_bot.regime.regime_classifier import classify_regime


def test_classify_regime_returns_unknown_for_short_df():
    data = {
        "open": list(range(10)),
        "high": list(range(1, 11)),
        "low": list(range(10)),
        "close": list(range(10)),
        "volume": [100] * 10,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df) == "unknown"
