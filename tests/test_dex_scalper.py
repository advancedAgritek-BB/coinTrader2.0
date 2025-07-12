import pandas as pd

import crypto_bot.strategy.dex_scalper as dex_scalper


def _make_df(prices):
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
        }
    )


def test_scalper_long_signal():
    close = pd.Series(range(1, 41))
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'long'
    assert 0 < score <= 1


def test_scalper_short_signal():
    close = pd.Series(range(40, 0, -1))
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'short'
    assert 0 < score <= 1


def test_scalper_neutral_signal():
    close = pd.Series([100.0] * 30)
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'none'
    assert score == 0.0


def test_scalper_min_data():
    close = pd.Series([100.0] * 10)
    df = pd.DataFrame({'close': close})
    score, direction = dex_scalper.generate_signal(df)
    assert direction == 'none'
    assert score == 0.0


def test_scalper_custom_config():
    close = pd.Series(range(1, 41))
    df = pd.DataFrame({'close': close})
    cfg = {'dex_scalper': {'ema_fast': 3, 'ema_slow': 10, 'min_signal_score': 0.05}}
    score, direction = dex_scalper.generate_signal(df, cfg)
    assert direction == 'long'
    assert score > 0


def test_atr_filter_blocks_signal():
    prices = list(range(1, 41))
    df = _make_df(prices)
    cfg = {"dex_scalper": {"min_atr_pct": 0.2}}
    score, direction = dex_scalper.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_atr_filter_allows_signal():
    prices = list(range(1, 41))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 10 for p in prices],
            "low": [p - 10 for p in prices],
            "close": prices,
        }
    )
    cfg = {"dex_scalper": {"min_atr_pct": 0.2}}
    score, direction = dex_scalper.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0
def test_scalper_fast_window_longer_than_df():
    """Ensure short DataFrame returns neutral when below required window."""
    close = pd.Series(range(20))
    df = pd.DataFrame({'close': close})
    cfg = {'dex_scalper': {'ema_fast': 30, 'ema_slow': 10}}
    score, direction = dex_scalper.generate_signal(df, cfg)
    assert direction == 'none'
    assert score == 0.0
