import pandas as pd
from frontend import utils


def test_compute_performance_short_then_cover():
    df = pd.DataFrame([
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 100},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.5, "price": 90},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.5, "price": 110},
    ])
    perf = utils.compute_performance(df)
    assert perf["XBT/USDT"] == 0


def test_compute_performance_flip_long_to_short():
    df = pd.DataFrame([
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 2, "price": 110},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100},
    ])
    perf = utils.compute_performance(df)
    assert perf["XBT/USDT"] == 20
