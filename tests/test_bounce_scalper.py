import pandas as pd
import pytest
import types
import sys
sys.modules.setdefault(
    "crypto_bot.strategy.sniper_bot", types.ModuleType("crypto_bot.strategy.sniper_bot")
)
import crypto_bot.strategy.bounce_scalper as bounce_scalper
from crypto_bot.strategy.bounce_scalper import BounceScalperConfig
from crypto_bot.cooldown_manager import configure, cooldowns


def _df(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )


def test_long_bounce_signal(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.5],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 500],
        }
    )
    monkeypatch.setattr(
        bounce_scalper.ta.trend,
        "ema_indicator",
        lambda s, window=50: pd.Series([4] * len(s)),
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "zscore_threshold": 1.0,
        "rsi_oversold_pct": 100,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0


def test_short_bounce_signal(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [4.5, 4.7, 4.8, 5.1],
            "high": [4.7, 5.0, 5.3, 5.2],
            "low": [4.3, 4.6, 4.7, 4.6],
            "close": [4.5, 4.8, 5.0, 4.6],
            "volume": [100, 100, 100, 500],
        }
    )
    monkeypatch.setattr(
        bounce_scalper.ta.trend,
        "ema_indicator",
        lambda s, window=50: pd.Series([6] * len(s)),
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "zscore_threshold": 1.0,
        "rsi_overbought_pct": 0,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "short"
    assert score > 0


def test_no_signal_without_volume_spike():
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.5],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 100],
        }
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "none"
    assert score == 0.0


def test_respects_cooldown(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(
        symbol="XBT/USDT",
        cooldown_enabled=True,
        oversold=30,
        overbought=70,
        volume_multiple=2.0,
    )
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0

    from crypto_bot import cooldown_manager

    cooldown_manager.configure(10)
    cooldown_manager.cooldowns.clear()
    cooldown_manager.mark_cooldown("BTC/USD", "bounce_scalper")

    score, direction = bounce_scalper.generate_signal(
        df, {"symbol": "BTC/USD", "cooldown_enabled": True}
    )
    assert direction == "none"
    assert score == 0.0


def test_mark_cooldown_called(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(
        symbol="XBT/USDT",
        cooldown_enabled=True,
        oversold=30,
        overbought=70,
        volume_multiple=2.0,
    )
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "short"
    assert score > 0

    called = {}

    def fake_mark(symbol, strat):
        called["symbol"] = symbol
        called["strategy"] = strat

    monkeypatch.setattr(bounce_scalper.cooldown_manager, "mark_cooldown", fake_mark)

    score, direction = bounce_scalper.generate_signal(df, config={"symbol": "ETH/USD"})
    assert direction == "long"
    assert score > 0.0
    assert called == {"symbol": "ETH/USD", "strategy": "bounce_scalper"}


def test_order_book_imbalance_blocks(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(
        symbol="XBT/USDT",
        cooldown_enabled=True,
        oversold=30,
        overbought=70,
        volume_multiple=2.0,
    )

    snap = {
        "type": "snapshot",
        "bids": [[30000.1, 1.0]],
        "asks": [[30000.2, 5.0]],
    }

    cfg = {"symbol": "BTC/USD", "imbalance_ratio": 2.0}
    score, direction = bounce_scalper.generate_signal(df, config=cfg, book=snap)
    assert direction == "none"
    assert score == 0.0


def test_order_book_imbalance_blocks_short(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    cfg = BounceScalperConfig(
        symbol="XBT/USDT",
        cooldown_enabled=True,
        oversold=30,
        overbought=70,
        volume_multiple=2.0,
    )

    snap = {
        "type": "snapshot",
        "bids": [[30000.1, 5.0]],
        "asks": [[30000.2, 1.0]],
    }

    cfg = {"symbol": "BTC/USD", "imbalance_ratio": 2.0}
    score, direction = bounce_scalper.generate_signal(df, config=cfg, book=snap)
    assert direction == "none"
    assert score == 0.0


def test_imbalance_penalty_reduces_score(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100] * 20 + [300]
    df = _df(prices, volumes)
    base_score, direction = bounce_scalper.generate_signal(
        df,
        BounceScalperConfig(
            symbol="XBT/USDT",
            oversold=30,
            overbought=70,
            volume_multiple=2.0,
        ),
    )
    assert direction == "long"

    snap = {
        "type": "snapshot",
        "bids": [[30000.1, 1.0]],
        "asks": [[30000.2, 5.0]],
    }

    cfg = {"symbol": "BTC/USD", "imbalance_ratio": 2.0, "imbalance_penalty": 0.5}
    score, direction = bounce_scalper.generate_signal(df, config=cfg, book=snap)
    assert direction == "long"
    assert 0 < score < base_score


def test_trend_filter_blocks_signals(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    # Force EMA above the last close so long signal should be blocked
    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([90]*len(s)))
    score, direction = bounce_scalper.generate_signal(
        df,
        BounceScalperConfig(symbol="XBT/USDT", oversold=30, overbought=70, volume_multiple=2.0),
    )
    assert direction == "none"
    assert score == 0.0


def test_dynamic_rsi_thresholds(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [300]
    df = _df(prices, volumes)

    # Mock ATR high enough to widen thresholds
    monkeypatch.setattr(
        bounce_scalper.ta.volatility,
        "average_true_range",
        lambda h, l, c, window=14: pd.Series([10.0]*len(c)),
    )

    # RSI just at default oversold
    monkeypatch.setattr(
        bounce_scalper.ta.momentum,
        "rsi",
        lambda s, window=14: pd.Series([30.0]*len(s)),
    )

    score, direction = bounce_scalper.generate_signal(
        df,
        BounceScalperConfig(symbol="XBT/USDT", oversold=30, overbought=70, volume_multiple=2.0),
    )
    # Threshold should drop below 30, preventing a signal
    assert direction == "none"
    assert score == 0.0


def test_volume_zscore_spike(monkeypatch):
    prices = list(range(100, 80, -1)) + [82]
    # create increasing volume to give non-zero std
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    # Provide realistic EMA below last close
    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([80]*len(s)))

    score, direction = bounce_scalper.generate_signal(
        df,
        BounceScalperConfig(symbol="XBT/USDT", oversold=30, overbought=70, volume_multiple=2.0),
    )
    assert direction == "long"
    assert score > 0


def test_cooldown_blocks_successive_signals(monkeypatch):
    configure(60)
    cooldowns.clear()
    prices = list(range(100, 80, -1)) + [82]
    volumes = [100 + i for i in range(20)] + [200]
    df = _df(prices, volumes)

    monkeypatch.setattr(bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([80]*len(s)))
    cfg = BounceScalperConfig(
        symbol="XBT/USDT",
        cooldown_enabled=True,
        oversold=30,
        overbought=70,
        volume_multiple=2.0,
    )
    first = bounce_scalper.generate_signal(df, config=cfg)
    second = bounce_scalper.generate_signal(df, config=cfg)

    assert first[1] != "none"
    assert second[1] == "none"


def test_config_from_dict():
    cfg = BounceScalperConfig.from_dict({"rsi_window": 10, "symbol": "ETH/USDT"})
    assert cfg.rsi_window == 10
    assert cfg.symbol == "ETH/USDT"
    # ensure defaults applied
    assert cfg.overbought == 65


def test_adaptive_rsi_threshold(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.5],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 500],
        }
    )

    monkeypatch.setattr(
        bounce_scalper.stats,
        "zscore",
        lambda s, lookback=3: pd.Series([0, 0, 0, -2], index=s.index),
    )

    cfg = {
        "rsi_window": 3,
        "vol_window": 3,
        "down_candles": 1,
        "up_candles": 2,
        "lookback": 3,
        "rsi_oversold_pct": 10,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, config=cfg)
    assert direction == "long"
    assert score > 0


def test_trigger_once(monkeypatch):
    configure(60)
    cooldowns.clear()
    prices = list(range(100, 40, -1)) + [42]
    volumes = [100 + i for i in range(60)] + [500]
    df = _df(prices, volumes)

    monkeypatch.setattr(
        bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([0] * len(s))
    )
    monkeypatch.setattr(
        bounce_scalper.ta.momentum, "rsi", lambda s, window=14: pd.Series([10] * len(s))
    )
    monkeypatch.setattr(
        bounce_scalper.stats, "zscore", lambda s, lookback: pd.Series([float("nan")] * len(s))
    )
    monkeypatch.setattr(
        bounce_scalper.ta.volatility,
        "average_true_range",
        lambda h, l, c, window: pd.Series([1] * len(h)),
    )
    monkeypatch.setattr(bounce_scalper, "is_engulfing", lambda df, body_pct: "bullish")
    monkeypatch.setattr(bounce_scalper, "confirm_higher_lows", lambda df, bars: True)
    monkeypatch.setattr(
        bounce_scalper, "normalize_score_by_volatility", lambda df, score: score
    )
    cfg = BounceScalperConfig(symbol="XBT/USDT", cooldown_enabled=True)
    first = bounce_scalper.generate_signal(df, config=cfg)
    second = bounce_scalper.generate_signal(df, config=cfg)
    third = bounce_scalper.generate_signal(df, config=cfg, force=True)
    fourth = bounce_scalper.generate_signal(df, config=cfg)

    assert first[1] != "none"
    assert second[1] == "none"
    assert third[1] != "none"
    assert fourth[1] == "none"


def test_trainer_model_influence(monkeypatch):
    df = _df([5.0, 5.0, 4.8, 4.5], [100, 100, 100, 500])
    monkeypatch.setattr(
        bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([4] * len(s))
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "zscore_threshold": 1.0,
        "rsi_oversold_pct": 100,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
        "atr_normalization": False,
    }
    monkeypatch.setattr(bounce_scalper, "MODEL", None)
    base, direction = bounce_scalper.generate_signal(df, config=cfg)
    dummy = types.SimpleNamespace(predict=lambda _df: 0.6)
    monkeypatch.setattr(bounce_scalper, "MODEL", dummy)
    score, direction2 = bounce_scalper.generate_signal(df, config=cfg)
    assert direction2 == direction
    assert score == pytest.approx((base + 0.6) / 2)


def test_lower_df_pattern_detection(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [5.0, 5.0, 4.8, 4.7],
            "high": [5.2, 5.1, 4.9, 5.2],
            "low": [4.8, 4.8, 4.5, 4.6],
            "close": [5.0, 4.8, 4.6, 5.0],
            "volume": [100, 100, 100, 500],
        }
    )
    lower_df = pd.DataFrame(
        {
            "open": [4.8, 4.5],
            "high": [5.0, 5.2],
            "low": [4.5, 4.4],
            "close": [4.6, 5.0],
            "volume": [100, 500],
        }
    )

    monkeypatch.setattr(
        bounce_scalper.ta.trend, "ema_indicator", lambda s, window=50: pd.Series([4] * len(s))
    )
    cfg = {
        "rsi_window": 3,
        "oversold": 100,
        "overbought": 0,
        "vol_window": 3,
        "zscore_threshold": 1.0,
        "rsi_oversold_pct": 100,
        "down_candles": 1,
        "up_candles": 2,
        "body_pct": 0.5,
    }
    score, direction = bounce_scalper.generate_signal(df, config=cfg, lower_df=lower_df)
    assert direction == "long"
    assert score > 0


def test_short_df_returns_none():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    score, direction = bounce_scalper.generate_signal(df, config={})
    assert direction == "none"
    assert score == 0.0

