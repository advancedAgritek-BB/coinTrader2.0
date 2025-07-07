import pandas as pd

from crypto_bot.strategy import bounce_scalper


def test_long_bounce_signal():
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.5],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 500],
        }
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_short_bounce_signal():
    df = pd.DataFrame(
        {
            "open": [4.5, 4.7, 4.8, 5.1],
            "high": [4.7, 5.0, 5.3, 5.2],
            "low": [4.3, 4.6, 4.7, 4.6],
            "close": [4.5, 4.8, 5.0, 4.6],
            "volume": [100, 100, 100, 500],
        }
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_no_signal_without_volume_spike():
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.5],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 100],
        }
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0
