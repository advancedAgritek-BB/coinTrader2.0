import pytest


from crypto_bot.paper_wallet import PaperWallet


def test_long_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("BTC/USDT", "buy", 1.0, 100.0)
    assert wallet.balance == 900.0

    pnl = wallet.close("BTC/USDT", 1.0, 110.0)
    tid = wallet.open("buy", 1.0, 100.0)
    assert wallet.balance == 910.0

    pnl = wallet.close(1.0, 110.0, tid)
    assert pnl == 10.0
    assert wallet.realized_pnl == 20.0
    assert wallet.balance == 1020.0


def test_short_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("BTC/USDT", "sell", 1.0, 100.0)
    assert wallet.balance == 1100.0

    pnl = wallet.close("BTC/USDT", 1.0, 90.0)
    tid = wallet.open("sell", 1.0, 100.0)
    assert wallet.balance == 1110.0

    pnl = wallet.close(1.0, 90.0, tid)
    assert pnl == 10.0
    assert wallet.realized_pnl == 20.0
    assert wallet.balance == 1020.0


def test_partial_closes_accumulate_pnl():
    wallet = PaperWallet(1000.0)
    wallet.open("BTC/USDT", "buy", 2.0, 50.0)
    assert wallet.balance == 900.0
    assert wallet.positions["BTC/USDT"]["size"] == 2.0

    pnl1 = wallet.close("BTC/USDT", 1.0, 55.0)
    assert pnl1 == 5.0
    assert wallet.realized_pnl == 5.0
    assert wallet.balance == 955.0
    assert wallet.positions["BTC/USDT"]["size"] == 1.0

    pnl2 = wallet.close("BTC/USDT", 1.0, 60.0)
    assert pnl2 == 10.0
    assert wallet.realized_pnl == 15.0
    assert wallet.balance == 1015.0
    assert "BTC/USDT" not in wallet.positions
def test_open_while_position_active_raises():
    wallet = PaperWallet(1000.0)
    wallet.open("BTC/USDT", "buy", 1.0, 10.0)
    with pytest.raises(RuntimeError):
        wallet.open("BTC/USDT", "buy", 1.0, 10.0)
    tid = wallet.open("buy", 2.0, 50.0)
    assert wallet.balance == 890.0
    assert wallet.positions[tid]["amount"] == 2.0

    pnl1 = wallet.close(1.0, 55.0, tid)
    assert pnl1 == 5.0
    assert wallet.realized_pnl == 5.0
    assert wallet.balance == 945.0
    assert wallet.positions[tid]["amount"] == 1.0

    pnl2 = wallet.close(1.0, 60.0, tid)
    assert pnl2 == 10.0
    assert wallet.realized_pnl == 15.0
    assert wallet.balance == 1005.0
    assert tid not in wallet.positions


def test_open_allows_multiple_positions():
    wallet = PaperWallet(1000.0)
    wallet.open("BTC/USDT", "buy", 1.0, 10.0)
    wallet.close("BTC/USDT", 1.0, 10.0)
    wallet.open("BTC/USDT", "sell", 1.0, 10.0)
    assert wallet.positions["BTC/USDT"]["size"] == 1.0
    id1 = wallet.open("buy", 1.0, 10.0, "t1")
    id2 = wallet.open("sell", 1.0, 10.0, "t2")
    assert set(wallet.positions.keys()) == {"BTC/USDT", "t1", "t2"}


def test_multiple_positions_unrealized_and_close():
    wallet = PaperWallet(1000.0)
    id1 = wallet.open("buy", 1.0, 100.0, "long1")
    id2 = wallet.open("sell", 2.0, 50.0, "short1")
    assert wallet.balance == 1000.0

    unreal = wallet.unrealized({"long1": 110.0, "short1": 40.0})
    assert unreal == 30.0


def test_open_multiple_positions_allowed():
    wallet = PaperWallet(1000.0, max_open_trades=2)
    id1 = wallet.open("buy", 1.0, 100.0, "long1")
    id2 = wallet.open("sell", 2.0, 50.0, "short1")
    assert wallet.position_size == 3.0

    pnl1 = wallet.close(1.0, 110.0, id1)
    pnl2 = wallet.close(2.0, 40.0, id2)
    assert wallet.realized_pnl == pnl1 + pnl2 == 30.0
    assert wallet.balance == 1030.0
    assert wallet.positions == {}
