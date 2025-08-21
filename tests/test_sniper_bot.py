import pandas as pd
import pytest
import types
import logging

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
    score, direction, _, event = sniper_bot.generate_signal(df, config=config)
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
        df, config={"symbol": "XRP/USD"}
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
        df, config={"price_fallback": True}
    )
    assert direction == "long"
    assert score > 0
    assert atr > 0
    assert event


def test_trainer_model_influence(monkeypatch):
    df = _df_with_volume_and_price(
        [1.0, 1.05, 1.1, 1.2],
        [10, 12, 11, 200]
    )
    cfg = {"atr_normalization": False}
    monkeypatch.setattr(sniper_bot, "MODEL", None)
    base, direction, _, _ = sniper_bot.generate_signal(df, config=cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.5)
    monkeypatch.setattr(sniper_bot, "MODEL", dummy)
    score, direction2, _, _ = sniper_bot.generate_signal(df, config=cfg)
    assert direction2 == direction
    assert score == pytest.approx((base + 0.5) / 2)


def test_logs_unknown_symbol(caplog):
    df = pd.DataFrame({
        "open": [1],
        "high": [1],
        "low": [1],
        "close": [1],
        "volume": [0],
    })
    with caplog.at_level(logging.INFO):
        sniper_bot.generate_signal(df)
    assert any(
        "Signal for unknown: 0.0, none" in r.getMessage() for r in caplog.records
    )


def test_zero_first_price_returns_none():
    import importlib, sys
    sys.modules.pop("crypto_bot.strategy.sniper_bot", None)
    real_sniper = importlib.import_module("crypto_bot.strategy.sniper_bot")
    df = _df_with_volume_and_price([0.0, 0.1, 0.2], [10, 20, 30])
    score, direction, _, _ = real_sniper.generate_signal(df)
    assert (score, direction) == (0.0, "none")
