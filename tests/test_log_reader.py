import pandas as pd
from crypto_bot import log_reader


def test_trade_summary(tmp_path):
    path = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 110},
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
        {"symbol": "XBT/USDT", "side": "buy", "amount": "x", "price": 100},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": "bad"},
    ]
    pd.DataFrame(data).to_csv(path, index=False, header=False)

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 2
    assert stats["total_pnl"] == 0


def test_trade_summary_no_history_defaults_win_rate(tmp_path):
    """When no trades are present the win rate should fall back to 0.6."""
    path = tmp_path / "trades.csv"

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 0
    assert stats["total_pnl"] == 0
    assert stats["win_rate"] == 0.6


def test_trade_summary_short_positions(tmp_path):
    path = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 100},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.5, "price": 90},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.5, "price": 110},
    ]
    pd.DataFrame(data).to_csv(path, index=False, header=False)

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 3
    assert stats["total_pnl"] == 0
    assert stats["win_rate"] == 0.5
    assert stats["active_positions"] == 0


def test_trade_summary_closes_and_opens_short(tmp_path):
    path = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 2, "price": 110},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100},
    ]
    pd.DataFrame(data).to_csv(path, index=False, header=False)

    stats = log_reader.trade_summary(path)
    assert stats["num_trades"] == 3
    assert stats["total_pnl"] == 20
    assert stats["win_rate"] == 1.0
    assert stats["active_positions"] == 0

