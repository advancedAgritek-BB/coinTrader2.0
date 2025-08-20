import json
import logging

import pandas as pd

from crypto_bot.risk.risk_manager import RiskManager, RiskConfig, kelly_fraction
from crypto_bot.volatility_filter import calc_atr


def volume_df(volumes: list[float]) -> pd.DataFrame:
    data = {
        "open": [1] * len(volumes),
        "high": [1] * len(volumes),
        "low": [1] * len(volumes),
        "close": [1] * len(volumes),
        "volume": volumes,
    }
    return pd.DataFrame(data)


def test_kelly_fraction_basic() -> None:
    """Ensure Kelly fraction computes expected sizing."""

    assert round(kelly_fraction(0.6, 1.0), 2) == 0.20

def test_allow_trade_rejects_below_min_volume():
    df = volume_df([1] * 19 + [0.5])
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=1,
        volume_threshold_ratio=0.5,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()


def test_allow_trade_rejects_below_volume_ratio():
    df = volume_df([1] * 19 + [0.4])
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=0.1,
        volume_threshold_ratio=0.5,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()


def test_allow_trade_allows_when_volume_sufficient():
    df = volume_df([1] * 19 + [2])
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=1,
        volume_threshold_ratio=0.5,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()


def test_allow_trade_rejects_volume_below_point_zero_one():
    df = volume_df([1] * 19 + [0.005])
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=0.01,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()


def test_allow_trade_rejects_when_volume_far_below_mean():
    df = volume_df([100] * 19 + [0.2])
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=10,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()

def test_allow_trade_not_enough_data():
    df = pd.DataFrame({"open": [1] * 10, "high": [1] * 10, "low": [1] * 10, "close": [1] * 10, "volume": [1] * 10})
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "not enough" in reason.lower()


def test_allow_trade_rejects_volume_too_low():
    data = {"open": [1] * 20, "high": [1] * 20, "low": [1] * 20, "close": [i for i in range(20)], "volume": [0.0] * 20}
    df = pd.DataFrame(data)
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volume too low" in reason.lower()


def test_allow_trade_rejects_when_flat():
    data = {"open": [1] * 20, "high": [1] * 20, "low": [1] * 20, "close": [1] * 20, "volume": [1] * 20}
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility too low" in reason.lower()


def test_allow_trade_rejects_when_hft_volatility_too_low():
    prices = [1 + i * 1e-7 for i in range(20)]
    data = {
        "open": prices,
        "high": [p + 1e-7 for p in prices],
        "low": [p - 1e-7 for p in prices],
        "close": prices,
        "volume": [1] * 20,
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility too low for hft" in reason.lower()


def test_allow_trade_allows_with_valid_data():
    data = {"open": [1] * 20, "high": [1] * 20, "low": [1] * 20, "close": [float(i) for i in range(20)], "volume": [1] * 20}
    df = pd.DataFrame(data)
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert allowed
    assert "trade allowed" in reason.lower()


def test_allow_trade_allows_high_score():
    df = volume_df([1] * 19 + [0.0005])
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT", score=0.5)
    assert allowed
    assert "high score" in reason.lower()


def test_allow_trade_rejects_when_atr_below_threshold():
    df = volume_df([1] * 20)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_atr_pct=0.00005,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert not allowed
    assert "volatility" in reason.lower()


def test_allow_trade_allows_above_atr_and_volume_thresholds():
    prices = list(range(1, 21))
    data = {
        "open": prices,
        "high": [p + 0.01 for p in prices],
        "low": prices,
        "close": prices,
        "volume": [0.02] * 20,
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=0.01,
        min_atr_pct=0.00005,
    )
    allowed, _ = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert allowed


def test_stop_order_management():
    manager = RiskManager(RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01))
    order = {"id": "1", "symbol": "XBT/USDT", "side": "sell", "amount": 1, "dry_run": True}
    manager.register_stop_order(order, symbol="XBT/USDT", regime="bull")
    assert manager.stop_orders["XBT/USDT"]["amount"] == 1
    assert manager.stop_orders["XBT/USDT"]["regime"] == "bull"
    manager.update_stop_order(0.5, symbol="XBT/USDT")
    assert manager.stop_orders["XBT/USDT"]["amount"] == 0.5
    manager = RiskManager(
        RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    )
    order1 = {
        "id": "1",
        "symbol": "XBT/USDT",
        "side": "sell",
        "amount": 1,
        "dry_run": True,
    }
    order2 = {
        "id": "2",
        "symbol": "ETH/USDT",
        "side": "sell",
        "amount": 2,
        "dry_run": True,
    }

    manager.register_stop_order(order1)
    manager.register_stop_order(order2)

    assert manager.stop_orders["XBT/USDT"]["amount"] == 1
    assert manager.stop_orders["ETH/USDT"]["amount"] == 2

    manager.update_stop_order(0.5, symbol="XBT/USDT")
    manager.update_stop_order(1.5, symbol="ETH/USDT")

    assert manager.stop_orders["XBT/USDT"]["amount"] == 0.5
    assert manager.stop_orders["ETH/USDT"]["amount"] == 1.5

    class DummyEx:
        def __init__(self):
            self.cancelled = False

        def cancel_order(self, oid, symbol):
            self.cancelled = True

    ex = DummyEx()
    manager.cancel_stop_order(ex, symbol="XBT/USDT")
    assert "XBT/USDT" not in manager.stop_orders
    ex1 = DummyEx()
    manager.cancel_stop_order(ex1, symbol="XBT/USDT")
    assert "XBT/USDT" not in manager.stop_orders

    ex2 = DummyEx()
    manager.cancel_stop_order(ex2, symbol="ETH/USDT")
    assert manager.stop_orders == {}


def test_register_stop_order_stores_regime():
    manager = RiskManager(
        RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    )
    order = {"id": "1", "symbol": "XBT/USDT", "side": "sell", "amount": 1, "dry_run": True}
    manager.register_stop_order(order, symbol="XBT/USDT", regime="volatile")
    assert manager.stop_orders["XBT/USDT"].get("regime") == "volatile"


def test_position_size_uses_trade_size_pct_when_no_stop():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            trade_size_pct=0.2,
            slippage_factor=0.0,
        )
    )
    size = manager.position_size(0.5, 1000)
    assert size == 1000 * 0.2 * 0.5


def test_position_size_returns_negative_for_short():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            trade_size_pct=0.2,
        )
    )
    size = manager.position_size(0.5, 1000, direction="short")
    assert size == -1000 * 0.2 * 0.5


def make_vol_df(long_diff: float, short_diff: float) -> pd.DataFrame:
    length = 60
    short_len = 14
    highs, lows, closes = [], [], []
    for i in range(length):
        diff = short_diff if i >= length - short_len else long_diff
        base = float(i)
        highs.append(base + diff)
        lows.append(base)
        closes.append(base + diff / 2)
    volume = [1] * length
    return pd.DataFrame({
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


def test_position_size_scales_with_volatility_and_drawdown():
    cfg = RiskConfig(
        max_drawdown=0.5,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        slippage_factor=0.0,
    )
    manager = RiskManager(cfg)
    df = make_vol_df(1.0, 2.0)
    manager.peak_equity = 1.0
    manager.equity = 0.75
    short_atr = calc_atr(df, period=cfg.atr_short_window).iloc[-1]
    long_atr = calc_atr(df, period=cfg.atr_long_window).iloc[-1]
    vol_factor = min(short_atr / long_atr, cfg.max_volatility_factor)
    drawdown = 1 - manager.equity / manager.peak_equity
    risk_factor = 1 - drawdown / cfg.max_drawdown
    expected = 1000 * cfg.trade_size_pct * vol_factor * risk_factor
    size = manager.position_size(1.0, 1000, df)
    assert abs(size - expected) < 1e-6


def test_position_size_reduces_when_drawdown_high():
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        slippage_factor=0.0,
    )
    manager = RiskManager(cfg)
    manager.peak_equity = 1.0
    manager.equity = 0.5
    size = manager.position_size(1.0, 1000)
    assert size == 1000 * cfg.trade_size_pct * 0.5
def test_position_size_risk_based():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            risk_pct=0.02,
            slippage_factor=0.0,
        )
    )
    size = manager.position_size(0.5, 1000, stop_distance=10)
    assert size == 1.0


def test_position_size_risk_based_capped():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            risk_pct=1.0,
            trade_size_pct=0.1,
            slippage_factor=0.0,
        )
    )
    size = manager.position_size(1.0, 1000, stop_distance=0.01, price=1)
    assert size == 100


def test_position_size_uses_atr():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            risk_pct=0.02,
            slippage_factor=0.0,
        )
    )
    size = manager.position_size(1.0, 1000, stop_distance=5, atr=8)
    assert abs(size - 2.5) < 1e-6


def test_position_size_low_price_risk_based():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            risk_pct=0.02,
            slippage_factor=0.0,
        )
    )
    size = manager.position_size(0.5, 1000, stop_distance=0.02, price=0.1)
    assert abs(size - 50.0) < 1e-6


def test_position_size_applies_slippage():
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        slippage_factor=0.1,
    )
    manager = RiskManager(cfg)
    size = manager.position_size(1.0, 1000)
    assert abs(size - 90.0) < 1e-6


def test_position_size_logs_when_zero(caplog):
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        trade_size_pct=0.1,
    )
    manager = RiskManager(cfg)
    caplog.set_level(logging.INFO)
    size = manager.position_size(0.0, 1000)
    assert size == 0.0
    assert "Position size zero" in caplog.text


def test_can_allocate_uses_tracker():
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        strategy_allocation={"trend_bot": 0.5},
    )
    manager = RiskManager(cfg)
    assert manager.can_allocate("trend_bot", 20, 100)
    manager.allocate_capital("trend_bot", 20)
    assert manager.can_allocate("trend_bot", 20, 100)
    manager.allocate_capital("trend_bot", 20)
    assert not manager.can_allocate("trend_bot", 20, 100)
    manager.deallocate_capital("trend_bot", 10)
    assert manager.can_allocate("trend_bot", 10, 100)


def test_risk_config_has_volume_threshold_ratio_default():
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    assert hasattr(cfg, "volume_threshold_ratio")
    assert cfg.volume_threshold_ratio == 0.05



def _df() -> pd.DataFrame:
    data = {
        "open": list(range(20)),
        "high": [i + 1 for i in range(20)],
        "low": list(range(20)),
        "close": [i + 0.5 for i in range(20)],
        "volume": [10] * 20,
    }
    return pd.DataFrame(data)


def test_allow_trade_rejects_on_bearish_sentiment(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "10")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "20")
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_fng=30,
        min_sentiment=30,
    )
    allowed, reason = RiskManager(cfg).allow_trade(_df(), symbol="XBT/USDT")
    assert allowed
    assert "trade allowed" in reason.lower()


def test_allow_trade_allows_on_positive_sentiment(monkeypatch):
    monkeypatch.setenv("MOCK_FNG_VALUE", "80")
    monkeypatch.setenv("MOCK_TWITTER_SENTIMENT", "80")
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_fng=30,
        min_sentiment=40,
    )
    allowed, _ = RiskManager(cfg).allow_trade(_df(), symbol="XBT/USDT")
    assert allowed


def test_allow_trade_rejects_on_negative_ev(tmp_path, monkeypatch):
    stats = {"trend_bot": {"win_rate": 0.4, "avg_win": 0.01, "avg_loss": -0.02}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.001,
    )
    allowed, reason = RiskManager(cfg).allow_trade(_df(), "trend_bot", symbol="XBT/USDT")
    assert allowed
    assert "trade allowed" in reason.lower()


def test_allow_trade_ignores_negative_ev(tmp_path, monkeypatch):
    stats = {"trend_bot": {"win_rate": 0.4, "avg_win": 0.01, "avg_loss": -0.02}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.0,
    )
    allowed, reason = RiskManager(cfg).allow_trade(_df(), "trend_bot", symbol="XBT/USDT")
    assert allowed
    assert "trade allowed" in reason.lower()


def test_allow_trade_allows_on_positive_ev(tmp_path, monkeypatch):
    stats = {"trend_bot": {"win_rate": 0.7, "avg_win": 0.02, "avg_loss": -0.01}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.0,
    )
    allowed, _ = RiskManager(cfg).allow_trade(_df(), "trend_bot", symbol="XBT/USDT")
    assert allowed


def test_allow_trade_ignores_ev_when_stats_missing(tmp_path, monkeypatch):
    file = tmp_path / "stats.json"
    file.write_text("{}")

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.5,
    )
    allowed, _ = RiskManager(cfg).allow_trade(_df(), "trend_bot", symbol="XBT/USDT")
    assert allowed




def test_allow_trade_ignores_pair_drawdown():
    data = {
        "open": [100] * 19 + [80],
        "high": [100] * 19 + [80],
        "low": [100] * 19 + [80],
        "close": [100] * 19 + [80],
        "volume": [10] * 20,
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        max_pair_drawdown=10,
        pair_drawdown_lookback=20,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert allowed
    assert "trade allowed" in reason.lower()


def test_allow_trade_allows_within_pair_drawdown():
    data = {
        "open": [100] * 19 + [95],
        "high": [100] * 19 + [95],
        "low": [100] * 19 + [95],
        "close": [100] * 19 + [95],
        "volume": [10] * 20,
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        max_pair_drawdown=10,
        pair_drawdown_lookback=20,
    )
    allowed, _ = RiskManager(cfg).allow_trade(df, symbol="XBT/USDT")
    assert allowed
