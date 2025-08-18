import pandas as pd

from crypto_bot.data.ohlcv_storage import load_ohlcv, save_ohlcv


def _df(ts_list, values):
    return pd.DataFrame(
        {
            "timestamp": ts_list,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": values,
        }
    )


def test_save_and_load_sorted(tmp_path):
    base = 1_000_000_000_000  # ms epoch
    df = _df([base + 60_000, base], [2, 1])  # milliseconds and unsorted
    save_ohlcv("binance", "BTC/USDT", "1m", df, tmp_path)

    file_path = tmp_path / "binance" / "BTC-USDT_1m.parquet"
    assert file_path.exists()

    loaded = load_ohlcv("binance", "BTC/USDT", "1m", tmp_path)
    assert loaded is not None
    assert loaded["timestamp"].tolist() == [base // 1000, (base + 60_000) // 1000]


def test_save_merges_dropping_duplicates(tmp_path):
    df1 = _df([1000, 2000], [1, 2])
    df2 = _df([2000, 3000], [4, 5])

    save_ohlcv("binance", "BTC/USDT", "1m", df1, tmp_path)
    save_ohlcv("binance", "BTC/USDT", "1m", df2, tmp_path)

    loaded = load_ohlcv("binance", "BTC/USDT", "1m", tmp_path)
    assert loaded is not None
    assert loaded["timestamp"].tolist() == [1000, 2000, 3000]
    # ensure duplicate timestamp uses latest values
    assert loaded.loc[loaded["timestamp"] == 2000, "open"].iloc[0] == 4


def test_load_missing_returns_none(tmp_path):
    assert load_ohlcv("binance", "ETH/USDT", "1m", tmp_path) is None
