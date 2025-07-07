import pandas as pd
from crypto_bot.strategy import bounce_scalper
from crypto_bot.strategy.bounce_scalper import BounceScalperConfig
from crypto_bot.cooldown_manager import configure, cooldowns


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
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_short_bounce_signal():
    prices = list(range(80, 100)) + [98]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_no_signal_without_volume_spike():
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 21
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0


def test_trend_filter_blocks_signals(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    # Force EMA above the last close so long signal should be blocked
    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([90]*len(s)))
    score, direction = bounce_scalper.generate_signal(df, BounceScalperConfig(symbol="BTC/USDT"))
    assert direction == "none"
    assert score == 0.0


def test_dynamic_rsi_thresholds(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [300]
    df = _df(prices, volumes)

    # Mock ATR high enough to widen thresholds
    monkeypatch.setattr(
        bounce_scalper.ta.volatility,
        "average_true_range",
        lambda h, l, c, window=14: pd.Series([10.0]*len(c)),
    )

    # RSI just at default oversold
    monkeypatch.setattr(
        bounce_scalper.ta.momentum,
        "rsi",
        lambda s, window=14: pd.Series([30.0]*len(s)),
    )

    score, direction = bounce_scalper.generate_signal(df, BounceScalperConfig(symbol="BTC/USDT"))
    # Threshold should drop below 30, preventing a signal
    assert direction == "none"
    assert score == 0.0


def test_volume_zscore_spike(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    # create increasing volume to give non-zero std
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    # Provide realistic EMA below last close
    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([80]*len(s)))

    score, direction = bounce_scalper.generate_signal(df, BounceScalperConfig(symbol="BTC/USDT"))
    assert direction == "long"
    assert score > 0


def test_cooldown_blocks_successive_signals(monkeypatch):
    configure(60)
    cooldowns.clear()
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([80]*len(s)))
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    first = bounce_scalper.generate_signal(df, cfg)
    second = bounce_scalper.generate_signal(df, cfg)

    assert first[1] != "none"
    assert second[1] == "none"


def test_config_from_dict():
    cfg = BounceScalperConfig.from_dict({"rsi_window": 10, "symbol": "ETH/USDT"})
    assert cfg.rsi_window == 10
    assert cfg.symbol == "ETH/USDT"
    # ensure defaults applied
    assert cfg.overbought == 70
