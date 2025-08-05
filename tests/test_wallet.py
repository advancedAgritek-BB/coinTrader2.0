import pytest

from crypto_bot.wallet import Wallet


def test_buy_sell_and_pnl():
    wallet = Wallet(initial_balance=1000)

    wallet.buy("BTC", 1, 100)
    wallet.buy("BTC", 1, 200)

    assert wallet.balance == pytest.approx(700)
    assert wallet.positions["BTC"].qty == pytest.approx(2)
    assert wallet.positions["BTC"].avg_price == pytest.approx(150)

    pnl = wallet.sell("BTC", 1, 300)
    assert pnl == pytest.approx(150)
    assert wallet.balance == pytest.approx(1000)
    assert wallet.positions["BTC"].qty == pytest.approx(1)

    total_pnl = wallet.update_pnl("BTC", 400)
    assert total_pnl == pytest.approx(400)

    total_bal = wallet.total_balance({"BTC": 400})
    assert total_bal == pytest.approx(1400)

    wallet.sell("BTC", 1, 400)
    assert "BTC" not in wallet.positions
    assert wallet.update_pnl("BTC", 0) == pytest.approx(400)
    assert wallet.total_balance({}) == pytest.approx(1400)
