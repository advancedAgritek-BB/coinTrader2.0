from crypto_bot.capital_tracker import CapitalTracker


def test_allocation_limits():
    tracker = CapitalTracker({'trend_bot': 0.5})
    assert tracker.can_allocate('trend_bot', 40, 100)
    tracker.allocate('trend_bot', 40)
    assert not tracker.can_allocate('trend_bot', 20, 100)
    tracker.deallocate('trend_bot', 10)
    assert tracker.can_allocate('trend_bot', 20, 100)
