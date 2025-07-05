import pytest

from crypto_bot.paper_wallet import PaperWallet


def test_long_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("buy", 1.0, 100.0)
    assert wallet.balance == 900.0

    pnl = wallet.close(1.0, 110.0)
    assert pnl == 10.0
    assert wallet.realized_pnl == 10.0
    assert wallet.balance == 1010.0


def test_short_position_profit():
    wallet = PaperWallet(1000.0)
    wallet.open("sell", 1.0, 100.0)
    assert wallet.balance == 1100.0

    pnl = wallet.close(1.0, 90.0)
    assert pnl == 10.0
    assert wallet.realized_pnl == 10.0
    assert wallet.balance == 1010.0


def test_partial_closes_accumulate_pnl():
    wallet = PaperWallet(1000.0)
    wallet.open("buy", 2.0, 50.0)
    assert wallet.balance == 900.0
    assert wallet.position_size == 2.0

    pnl1 = wallet.close(1.0, 55.0)
    assert pnl1 == 5.0
    assert wallet.realized_pnl == 5.0
    assert wallet.balance == 955.0
    assert wallet.position_size == 1.0

    pnl2 = wallet.close(1.0, 60.0)
    assert pnl2 == 10.0
    assert wallet.realized_pnl == 15.0
    assert wallet.balance == 1015.0
    assert wallet.position_size == 0.0
    assert wallet.entry_price is None
    assert wallet.side is None
def test_open_while_position_active_raises():
    wallet = PaperWallet(100.0)
    wallet.open("buy", 1.0, 10.0)
    with pytest.raises(RuntimeError):
        wallet.open("buy", 1.0, 10.0)


def test_open_after_close_succeeds():
    wallet = PaperWallet(100.0)
    wallet.open("buy", 1.0, 10.0)
    wallet.close(1.0, 10.0)
    wallet.open("sell", 1.0, 10.0)
    assert wallet.position_size == 1.0

