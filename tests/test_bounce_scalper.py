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


def test_respects_cooldown(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)

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

    snap = {
        "type": "snapshot",
        "bids": [[30000.1, 1.0]],
        "asks": [[30000.2, 5.0]],
    }

    cfg = {"symbol": "BTC/USD", "order_book": snap, "imbalance_ratio": 2.0}
    score, direction = bounce_scalper.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0
