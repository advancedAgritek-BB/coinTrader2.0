import pandas as pd
from pathlib import Path

from crypto_bot.utils import open_trades


def test_open_trades_simple(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100, "timestamp": "t1"},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 110, "timestamp": "t2"},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1.5, "price": 120, "timestamp": "t3"},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert open_orders == [
        {
            "symbol": "XBT/USDT",
            "side": "long",
            "amount": 0.5,
            "price": 110.0,
            "entry_time": "t2",
        }
    ]


def test_open_trades_multiple_symbols(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "ETH/USDT", "side": "buy", "amount": 1, "price": 50, "timestamp": "a"},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 2, "price": 100, "timestamp": "b"},
        {"symbol": "ETH/USDT", "side": "sell", "amount": 0.5, "price": 60, "timestamp": "c"},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 110, "timestamp": "d"},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert sorted(open_orders, key=lambda x: (x["symbol"], x["entry_time"])) == [
        {
            "symbol": "XBT/USDT",
            "side": "long",
            "amount": 1.0,
            "price": 100.0,
            "entry_time": "b",
        },
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "amount": 0.5,
            "price": 50.0,
            "entry_time": "a",
        },
    ]


def test_open_trades_short_position(tmp_path: Path):
    """Short after sell then partial buy."""
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 100, "timestamp": "t1"},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.4, "price": 90, "timestamp": "t2"},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert open_orders == [
        {
            "symbol": "XBT/USDT",
            "side": "short",
            "amount": 0.6,
            "price": 100.0,
            "entry_time": "t1",
        }
    ]


def test_open_trades_short_entries(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 100, "timestamp": "a"},
        {"symbol": "XBT/USDT", "side": "buy", "amount": 0.4, "price": 95, "timestamp": "b"},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 0.5, "price": 105, "timestamp": "c"},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert open_orders == [
        {
            "symbol": "XBT/USDT",
            "side": "short",
            "amount": 0.6,
            "price": 100.0,
            "entry_time": "a",
        },
        {
            "symbol": "XBT/USDT",
            "side": "short",
            "amount": 0.5,
            "price": 105.0,
            "entry_time": "c",
        },
    ]


def test_open_trades_buy_closes_short(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "ETH/USDT", "side": "sell", "amount": 1, "price": 20, "timestamp": "a"},
        {"symbol": "ETH/USDT", "side": "buy", "amount": 1.5, "price": 18, "timestamp": "b"},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert open_orders == [
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "amount": 0.5,
            "price": 18.0,
            "entry_time": "b",
        }
    ]


def test_open_trades_ignore_stop_rows(tmp_path: Path):
    log_file = tmp_path / "trades.csv"
    data = [
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100, "timestamp": "t1", "is_stop": False},
        {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 0, "timestamp": "t2", "is_stop": True, "stop_price": 90},
    ]
    pd.DataFrame(data).to_csv(log_file, index=False, header=False)

    open_orders = open_trades.get_open_trades(log_file)
    assert open_orders == [
        {
            "symbol": "XBT/USDT",
            "side": "long",
            "amount": 1.0,
            "price": 100.0,
            "entry_time": "t1",
        }
    ]
