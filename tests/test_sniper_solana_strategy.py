import pandas as pd

from crypto_bot.strategy import sniper_solana


def make_df(prices):
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})


def test_skip_on_flags(monkeypatch):
    df = make_df([1.0, 1.0])
    score, direction = sniper_solana.generate_signal(df, {"is_trading": False})
    assert score == 0.0
    assert direction == "none"

    score, direction = sniper_solana.generate_signal(df, {"conf_pct": 0.6})
    assert score == 0.0
    assert direction == "none"


def test_uses_pyth_price(monkeypatch):
    df = make_df([1.0, 1.0])

    def fake_price(symbol, cfg=None):
        return 3.0

    monkeypatch.setattr(sniper_solana, "get_pyth_price", fake_price)
    cfg = {"token": "SOL", "atr_window": 1, "jump_mult": 1.0}
    score, direction = sniper_solana.generate_signal(df, cfg)
    assert direction == "long"
    assert score == 1.0
