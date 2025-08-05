import pytest
from crypto_bot.paper_wallet import PaperWallet


def test_buy_then_sell_realizes_pnl_and_restores_balance():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)
    pnl = wallet.close("XBT/USDT", 1.0, 110.0)
    assert pnl == 10.0
    assert wallet.realized_pnl == 10.0
    assert wallet.balance == 1010.0


def test_multiple_buys_adjust_average_entry():
    wallet = PaperWallet(1000.0)
    wallet.open("buy", 1.0, 100.0, "t1")
    wallet.open("buy", 2.0, 200.0, "t2")
    expected = (1.0 * 100.0 + 2.0 * 200.0) / 3.0
    assert wallet.entry_price == pytest.approx(expected)


def test_partial_sells_retain_remaining_position():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 10.0, 10.0)
    pnl = wallet.close("XBT/USDT", 4.0, 12.0)
    assert pnl == 8.0
    assert wallet.realized_pnl == 8.0
    assert wallet.positions["XBT/USDT"]["size"] == 6.0


def test_unrealized_pnl_update():
    wallet = PaperWallet(1000.0)
    tid = wallet.open("buy", 2.0, 100.0, "long")
    unreal = wallet.unrealized({"long": 110.0})
    assert unreal == 20.0


def test_total_balance_with_cash_and_positions():
    wallet = PaperWallet(1000.0)
    wallet.open("buy", 1.0, 100.0, "long")
    wallet.open("sell", 1.0, 50.0, "short")
    prices = {"long": 110.0, "short": 40.0}
    total = wallet.balance
    for pos in wallet.positions.values():
        qty = pos.get("size", pos.get("amount", 0.0))
        if pos["side"] == "buy":
            total += pos["entry_price"] * qty
        else:
            total += pos["reserved"]
    total += wallet.unrealized(prices)
    assert total == pytest.approx(1020.0)
