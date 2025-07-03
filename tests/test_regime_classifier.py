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


def test_classify_regime_returns_unknown_for_14_rows():
    data = {
        "open": list(range(14)),
        "high": list(range(1, 15)),
        "low": list(range(14)),
        "close": list(range(14)),
        "volume": [100] * 14,
    }
    df = pd.DataFrame(data)
    assert classify_regime(df) == "unknown"


def test_classify_regime_returns_unknown_between_15_and_19_rows():
    for rows in range(15, 20):
        data = {
            "open": list(range(rows)),
            "high": list(range(1, rows + 1)),
            "low": list(range(rows)),
            "close": list(range(rows)),
            "volume": [100] * rows,
        }
        df = pd.DataFrame(data)
        assert classify_regime(df) == "unknown"
