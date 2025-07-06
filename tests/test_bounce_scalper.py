import pandas as pd

from crypto_bot.strategy import bounce_scalper


def _df(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )


def test_long_bounce_signal():
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    score, direction = bounce_scalper.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_short_bounce_signal():
    prices = list(range(80, 100)) + [98]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    score, direction = bounce_scalper.generate_signal(df)
    assert direction == "short"
    assert score > 0


def test_no_signal_without_volume_spike():
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 21
    df = _df(prices, volumes)
    score, direction = bounce_scalper.generate_signal(df)
    assert direction == "none"
    assert score == 0.0
