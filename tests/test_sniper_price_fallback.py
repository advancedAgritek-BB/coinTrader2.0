import pandas as pd
from crypto_bot.strategy import sniper_bot


def make_df():
    bars = 15
    opens = [1.0] * bars
    highs = [1.05] * (bars - 1) + [1.25]
    lows = [0.95] * (bars - 1) + [1.0]
    closes = [1.0] * (bars - 1) + [1.25]
    volumes = [100] * (bars - 1) + [250]
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_price_fallback_long_signal():
    df = make_df()
    score, direction, atr, event = sniper_bot.generate_signal(
        df, {"price_fallback": True}
    )
    assert direction == "long"
    assert score > 0
    assert atr > 0
    assert event
