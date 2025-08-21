import pandas as pd
from crypto_bot.strategy import trend_bot


def test_normalized_score_lower_when_atr_low():
    # build data with high volatility then low volatility
    close = list(range(60))
    high = [c + (1 if i < 46 else 0.05) for i, c in enumerate(close)]
    low = [c - (1 if i < 46 else 0.05) for i, c in enumerate(close)]
    volume = [1] * 59 + [2]
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})

    cfg = {"atr_normalization": False, "donchian_confirmation": False}
    raw_score, _ = trend_bot.generate_signal(df, config=cfg)
    norm_score, _ = trend_bot.generate_signal(df, config={"donchian_confirmation": False})

    assert norm_score < raw_score
    assert raw_score > 0

