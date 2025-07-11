import pandas as pd
from datetime import timedelta
from crypto_bot import tax_logger


def test_tax_logger_gain_and_csv(tmp_path):
    tax_logger._open_positions.clear()
    tax_logger._closed_trades.clear()

    entry = {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 10000}
    tax_logger.record_entry(entry)
    # simulate 30 day holding period
    tax_logger._open_positions[0]["time"] -= timedelta(days=30)

    exit_order = {"symbol": "XBT/USDT", "side": "sell", "amount": 1, "price": 12000}
    tax_logger.record_exit(exit_order)

    trade = tax_logger._closed_trades[0]
    assert trade["Profit"] == 2000
    assert trade["Type"] == "short_term"

    out = tmp_path / "tax.csv"
    tax_logger.export_csv(out)
    df = pd.read_csv(out)
    assert df.iloc[0]["Profit"] == 2000
