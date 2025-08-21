from crypto_bot.strategy import flash_crash_bot
from crypto_bot import strategy_router
import pandas as pd


def test_get_strategy_by_name():
    fn = strategy_router.get_strategy_by_name("flash_crash_bot")
    assert callable(fn)


def _make_df(drop=True, volume_spike=True):
    base = [100.0] * 20
    last_price = 90.0 if drop else 95.0
    prices = base + [last_price]
    volumes = [100.0] * 20 + ([1000.0] if volume_spike else [400.0])
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )


def test_long_signal_on_flash_crash():
    df = _make_df(True, True)
    score, direction = flash_crash_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_no_signal_when_conditions_fail():
    df = _make_df(True, False)
    assert flash_crash_bot.generate_signal(df) == (0.0, "none")


def test_ema_filter_blocks():
    prices = [60.0] * 20 + [100.0, 90.0]
    volumes = [100.0] * 21 + [1000.0]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )
    cfg = {"flash_crash": {"ema_window": 5}}
    assert flash_crash_bot.generate_signal(df, config=cfg) == (0.0, "none")
