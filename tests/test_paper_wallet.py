import pytest

from crypto_bot.paper_wallet import PaperWallet


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

