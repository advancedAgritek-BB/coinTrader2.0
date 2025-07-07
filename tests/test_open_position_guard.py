from crypto_bot.open_position_guard import OpenPositionGuard


def test_guard_allows_when_below_limit():
    guard = OpenPositionGuard(2)
    positions = {"BTC/USDT": {"side": "buy"}}
    assert guard.can_open(positions)


def test_guard_rejects_at_limit():
    guard = OpenPositionGuard(1)
    positions = {"BTC/USDT": {"side": "buy"}}
    assert not guard.can_open(positions)
