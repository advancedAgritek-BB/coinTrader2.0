import pandas as pd

from crypto_bot.strategy import sniper_bot


def _df_with_volume_and_price(close_list, volume_list):
    return pd.DataFrame({
        "open": close_list,
        "high": close_list,
        "low": close_list,
        "close": close_list,
        "volume": volume_list,
    })


def test_sniper_triggers_on_breakout():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    score, direction = sniper_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.8


def test_sniper_ignores_low_volume():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [1, 1, 1, 2]
    )
    score, direction = sniper_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_sniper_respects_history_length():
    df = _df_with_volume_and_price([1.0] * 100, [10] * 100)
    score, direction = sniper_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_direction_override_short():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    config = {"direction": "short"}
    score, direction = sniper_bot.generate_signal(df, config)
    assert direction == "short"
    assert score > 0.8
