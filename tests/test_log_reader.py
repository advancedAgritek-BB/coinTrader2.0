import pandas as pd
from crypto_bot import log_reader


def test_trade_summary(tmp_path):
    path = tmp_path / "trades.csv"
    data = [
        {"symbol": "BTC/USDT", "side": "buy", "amount": 1, "price": 100},
        {"symbol": "BTC/USDT", "side": "sell", "amount": 1, "price": 110},
        {"symbol": "ETH/USDT", "side": "buy", "amount": 2, "price": 200},
    ]
    pd.DataFrame(data).to_csv(path, index=False, header=False)

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 3
    assert stats["total_pnl"] == 10
    assert stats["win_rate"] == 1.0
    assert stats["active_positions"] == 2


def test_trade_summary_handles_bad_rows(tmp_path):
    path = tmp_path / "trades.csv"
    data = [
        {"symbol": "BTC/USDT", "side": "buy", "amount": "x", "price": 100},
        {"symbol": "BTC/USDT", "side": "sell", "amount": 1, "price": "bad"},
    ]
    pd.DataFrame(data).to_csv(path, index=False, header=False)

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 2
    assert stats["total_pnl"] == 0

