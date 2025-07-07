import pandas as pd
from crypto_bot.strategy import bounce_scalper
from crypto_bot.strategy.bounce_scalper import BounceScalperConfig
from crypto_bot.cooldown_manager import configure, cooldowns


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


def test_respects_cooldown(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0

    from crypto_bot import cooldown_manager

    cooldown_manager.configure(10)
    cooldown_manager.cooldowns.clear()
    cooldown_manager.mark_cooldown("BTC/USD", "bounce_scalper")

    score, direction = bounce_scalper.generate_signal(df, {"symbol": "BTC/USD"})
    assert direction == "none"
    assert score == 0.0


def test_mark_cooldown_called(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(symbol="BTC/USDT")
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0

    called = {}

    def fake_mark(symbol, strat):
        called["symbol"] = symbol
        called["strategy"] = strat

    monkeypatch.setattr(bounce_scalper.cooldown_manager, "mark_cooldown", fake_mark)

    score, direction = bounce_scalper.generate_signal(df, {"symbol": "ETH/USD"})
    assert direction == "long"
    assert score > 0.0
    assert called == {"symbol": "ETH/USD", "strategy": "bounce_scalper"}


def test_order_book_imbalance_blocks(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(symbol="BTC/USDT")

    snap = {
        "type": "snapshot",
        "bids": [[30000.1, 1.0]],
        "asks": [[30000.2, 5.0]],
    }

    cfg = {"symbol": "BTC/USD", "order_book": snap, "imbalance_ratio": 2.0}
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
