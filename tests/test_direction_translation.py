import crypto_bot.main as main


def test_direction_to_side_and_opposite():
    assert main.direction_to_side("long") == "buy"
    assert main.direction_to_side("short") == "sell"
    assert main.direction_to_side("none") == "none"
    assert main.opposite_side("buy") == "sell"
    assert main.opposite_side("sell") == "buy"
