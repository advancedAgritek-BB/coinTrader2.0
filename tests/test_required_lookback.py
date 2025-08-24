import pytest

from crypto_bot.strategy import breakout_bot, mean_bot, trend_bot, registry
from crypto_bot import config


@pytest.mark.parametrize(
    "module,cfg_section,expected",
    [
        (
            breakout_bot,
            {
                "breakout_bot": {
                    "ema_window": 250,
                    "adx_window": 40,
                    "bb_length": 20,
                    "kc_length": 30,
                    "donchian_window": 120,
                    "volume_window": 25,
                }
            },
            250,
        ),
        (
            trend_bot,
            {
                "trend_bot": {
                    "indicator_lookback": 200,
                    "volume_window": 120,
                    "trend_ema_fast": 5,
                    "trend_ema_slow": 55,
                    "atr_period": 14,
                }
            },
            200,
        ),
        (
            mean_bot,
            {
                "mean_bot": {
                    "indicator_lookback": 90,
                    "adx_window": 40,
                }
            },
            90,
        ),
    ],
)
def test_compute_required_lookback_per_strategy(monkeypatch, module, cfg_section, expected):
    cfg = {"timeframes": ["1m"]}
    cfg.update(cfg_section)
    monkeypatch.setattr(config, "cfg", cfg)
    out = registry.compute_required_lookback_per_tf([module])
    assert out == {"1m": expected}


def test_compute_required_lookback_aggregate(monkeypatch):
    cfg = {
        "timeframes": ["1m"],
        "breakout_bot": {
            "ema_window": 250,
            "adx_window": 40,
            "bb_length": 20,
            "kc_length": 30,
            "donchian_window": 120,
            "volume_window": 25,
        },
        "trend_bot": {
            "indicator_lookback": 200,
            "volume_window": 120,
            "trend_ema_fast": 5,
            "trend_ema_slow": 55,
            "atr_period": 14,
        },
        "mean_bot": {"indicator_lookback": 90, "adx_window": 40},
    }
    monkeypatch.setattr(config, "cfg", cfg)
    out = registry.compute_required_lookback_per_tf([breakout_bot, trend_bot, mean_bot])
    assert out == {"1m": 250}

