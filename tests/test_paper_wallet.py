import pytest

from crypto_bot.paper_wallet import PaperWallet


def test_long_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)
    assert wallet.balance == 900.0

    pnl = wallet.close("XBT/USDT", 1.0, 110.0)
    assert pnl == 10.0
    assert wallet.realized_pnl == 10.0
    assert wallet.balance == 1010.0

    tid = wallet.open("buy", 1.0, 100.0)
    assert wallet.balance == 910.0

    pnl = wallet.close(1.0, 110.0, tid)
    assert pnl == 10.0
    assert wallet.realized_pnl == 20.0
    assert wallet.balance == 1020.0


def test_short_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "sell", 1.0, 100.0)
    assert wallet.balance == 900.0
    assert wallet.positions["XBT/USDT"]["reserved"] == 100.0

    pnl = wallet.close("XBT/USDT", 1.0, 90.0)
    assert pnl == 10.0
    assert wallet.realized_pnl == 10.0
    assert wallet.balance == 1010.0

    tid = wallet.open("sell", 1.0, 100.0)
    assert wallet.balance == 910.0
    assert wallet.positions[tid]["reserved"] == 100.0

    pnl = wallet.close(1.0, 90.0, tid)
    assert pnl == 10.0
    assert wallet.realized_pnl == 20.0
    assert wallet.balance == 1020.0


def test_partial_closes_accumulate_pnl():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 2.0, 50.0)
    assert wallet.balance == 900.0

    pnl1 = wallet.close("XBT/USDT", 1.0, 55.0)
    assert pnl1 == 5.0
    assert wallet.realized_pnl == 5.0
    assert wallet.balance == 955.0

    pnl2 = wallet.close("XBT/USDT", 1.0, 60.0)
    assert pnl2 == 10.0
    assert wallet.realized_pnl == 15.0
    assert wallet.balance == 1015.0
    assert "XBT/USDT" not in wallet.positions


def test_open_while_position_active_raises():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 10.0)
    with pytest.raises(RuntimeError):
        wallet.open("XBT/USDT", "buy", 1.0, 10.0)

    # Close the initial trade before opening a new one
    wallet.close("XBT/USDT", 1.0, 10.0)

    tid = wallet.open("buy", 2.0, 50.0)
    assert wallet.balance == 900.0
    assert wallet.positions[tid]["amount"] == 2.0

    pnl1 = wallet.close(1.0, 55.0, tid)
    assert pnl1 == 5.0
    assert wallet.realized_pnl == 5.0
    assert wallet.balance == 955.0
    assert wallet.positions[tid]["amount"] == 1.0

    pnl2 = wallet.close(1.0, 60.0, tid)
    assert pnl2 == 10.0
    assert wallet.realized_pnl == 15.0
    assert wallet.balance == 1015.0
    assert tid not in wallet.positions


def test_open_allows_multiple_positions():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 10.0)
    wallet.close("XBT/USDT", 1.0, 10.0)
    wallet.open("XBT/USDT", "sell", 1.0, 10.0)
    assert wallet.positions["XBT/USDT"]["size"] == 1.0
    id1 = wallet.open("buy", 1.0, 10.0, "t1")
    id2 = wallet.open("sell", 1.0, 10.0, "t2")
    assert set(wallet.positions.keys()) == {"XBT/USDT", "t1", "t2"}

    pnl1 = wallet.close("XBT/USDT", 1.0, 10.0)
    pnl2 = wallet.close(1.0, 10.0, id1)
    pnl3 = wallet.close(1.0, 10.0, id2)

    assert pnl1 == 0.0
    assert pnl2 == 0.0
    assert pnl3 == 0.0
    assert wallet.realized_pnl == 0.0
    assert wallet.balance == 1000.0
    assert wallet.positions == {}


def test_multiple_positions_unrealized_and_close():
    wallet = PaperWallet(1000.0)
    long_id = wallet.open("buy", 1.0, 100.0, "long1")
    short_id = wallet.open("sell", 2.0, 50.0, "short1")
    assert wallet.balance == 800.0

    unreal = wallet.unrealized({"long1": 110.0, "short1": 40.0})
    assert unreal == 30.0

    pnl_long = wallet.close(1.0, 110.0, long_id)
    pnl_short = wallet.close(2.0, 40.0, short_id)
    assert pnl_long == 10.0
    assert pnl_short == 20.0
    assert wallet.realized_pnl == 30.0
    assert wallet.balance == 1030.0
    assert wallet.positions == {}

def test_open_multiple_positions_allowed():
    wallet = PaperWallet(1000.0, max_open_trades=2)
    id1 = wallet.open("buy", 1.0, 100.0, "long1")
    id2 = wallet.open("sell", 2.0, 50.0, "short1")
    with pytest.raises(RuntimeError):
        wallet.open("buy", 1.0, 100.0, "long2")

    pnl1 = wallet.close(1.0, 110.0, id1)
    pnl2 = wallet.close(2.0, 40.0, id2)

    assert pnl1 == 10.0
    assert pnl2 == 20.0
    assert wallet.realized_pnl == 30.0
    assert wallet.balance == 1030.0
    assert wallet.positions == {}


def test_buy_rejected_when_insufficient_balance():
    wallet = PaperWallet(100.0)
    with pytest.raises(RuntimeError):
        wallet.open("buy", 2.0, 60.0)
    assert wallet.balance == 100.0
    assert wallet.positions == {}


def test_short_sell_rejected():
    wallet = PaperWallet(1000.0, short_selling=False)
    with pytest.raises(RuntimeError):
        wallet.open("XBT/USDT", "sell", 1.0, 100.0)


def test_close_falls_back_to_entry_price_long():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "buy", 1.0, 100.0)

    pnl = wallet.close("XBT/USDT", 1.0, 0.0)
    assert pnl == 0.0
    assert wallet.balance == 1000.0
    assert wallet.realized_pnl == 0.0


def test_close_falls_back_to_entry_price_short():
    wallet = PaperWallet(1000.0)
    wallet.open("XBT/USDT", "sell", 1.0, 100.0)

    pnl = wallet.close("XBT/USDT", 1.0, 0.0)
    assert pnl == 0.0
    assert wallet.balance == 1000.0
    assert wallet.realized_pnl == 0.0
