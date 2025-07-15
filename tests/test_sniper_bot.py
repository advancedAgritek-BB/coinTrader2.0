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
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.8
    assert not event


def test_sniper_ignores_low_volume():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [1, 1, 1, 2]
    )
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0
    assert not event


def test_sniper_respects_history_length():
    df = _df_with_volume_and_price([1.0] * 100, [10] * 100)
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0
    assert not event


def test_direction_override_short():
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    config = {"direction": "short"}
    score, direction, _, event = sniper_bot.generate_signal(df, config)
    assert direction == "short"
    assert score > 0.8
    assert not event


def test_auto_short_on_price_drop():
    df = _df_with_volume_and_price(
        [1.0, 0.95, 0.9, 0.8],
        [10, 12, 11, 200]
    )
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0.8
    assert not event


def test_high_freq_short_window():
    df = _df_with_volume_and_price(
        [1.0, 1.1, 1.2],
        [10, 12, 40]
    )
    score, direction, _, event = sniper_bot.generate_signal(
        df, high_freq=True, config={"min_volume": 1}
    )
    assert direction == "long"
    assert score > 0.8
    assert not event


def test_symbol_filter_blocks_disallowed():
    df = _df_with_volume_and_price(
        [1.0, 1.1, 1.2],
        [10, 12, 40]
    )
    score, direction, _, event = sniper_bot.generate_signal(
        df, {"symbol": "XRP/USD"}
    )
    assert direction == "none"
    assert score == 0.0
    assert not event


def test_event_trigger():
    df = pd.DataFrame({
        "open": [1, 1, 1, 1, 1],
        "high": [1.1, 1.1, 1.1, 1.1, 5],
        "low": [0.9, 0.9, 0.9, 0.9, 1],
        "close": [1, 1, 1, 1, 5],
        "volume": [10, 10, 10, 10, 50],
    })
    score, direction, _, event = sniper_bot.generate_signal(
        df, config={"atr_window": 4, "volume_window": 4, "min_volume": 1}
    )
    assert direction == "long"
    assert score > 0
    assert event


def test_defaults_trigger_on_small_breakout():
    df = _df_with_volume_and_price(
        [1.0, 1.0, 1.0, 1.06],
        [100, 100, 100, 160],
    )
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0
    assert not event


def test_fallback_short_no_breakout():
    df = _df_with_volume_and_price(
        [1.0, 0.99, 0.98, 0.97],
        [120, 110, 100, 130],
    )
    score, direction, _, event = sniper_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0
    assert not event
def test_price_fallback_long_signal():
    bars = 15
    df = pd.DataFrame({
        "open": [1.0] * bars,
        "high": [1.05] * (bars - 1) + [1.25],
        "low": [0.95] * (bars - 1) + [1.0],
        "close": [1.0] * (bars - 1) + [1.25],
        "volume": [100] * (bars - 1) + [250],
    })
    score, direction, atr, event = sniper_bot.generate_signal(
        df, {"price_fallback": True}
    )
    assert direction == "long"
    assert score > 0
    assert atr > 0
    assert event
