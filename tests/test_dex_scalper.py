import pandas as pd

from crypto_bot.strategy import dex_scalper


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


def test_scalper_blocks_disallowed_pair(monkeypatch):
    close = pd.Series(range(1, 41))
    df = pd.DataFrame({'close': close})
    monkeypatch.setattr(dex_scalper, 'ALLOWED_PAIRS', ['ETH/USD'])
    score, direction = dex_scalper.generate_signal(df, {'symbol': 'BTC/USD'})
    assert (score, direction) == (0.0, 'none')
