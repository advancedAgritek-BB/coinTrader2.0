import json
import pandas as pd
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.volatility_filter import calc_atr
from crypto_bot.utils import trade_memory
from crypto_bot.utils import ev_tracker
import logging

def test_allow_trade_rejects_low_volume():
    data = {
        'open':[1]*20,
        'high':[1]*20,
        'low':[1]*20,
        'close':[1]*20,
        'volume':[2]*19 + [0]
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=1,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "min volume" in reason


def test_allow_trade_respects_volume_threshold():
    data = {
        'open': [1] * 20,
        'high': [1] * 20,
        'low': [1] * 20,
        'close': [1] * 20,
        'volume': [1] * 19 + [2],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        volume_threshold_ratio=0.5,
    )
    allowed, _ = RiskManager(cfg).allow_trade(df)
    assert allowed


def test_allow_trade_allows_when_only_ratio_fails():
    data = {
        'open': [1] * 20,
        'high': [1] * 20,
        'low': [1] * 20,
        'close': [1] * 20,
        'volume': [10] * 19 + [4],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=3,
        volume_threshold_ratio=0.5,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "50% of mean" in reason


def test_allow_trade_allows_when_only_min_volume_fails():
    data = {
        'open': [1] * 20,
        'high': [1] * 20,
        'low': [1] * 20,
        'close': [1] * 20,
        'volume': [10] * 19 + [9],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=11,
        volume_threshold_ratio=0.5,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "min volume" in reason


def test_allow_trade_logs_ratio_reason():
    data = {
        'open': [1] * 20,
        'high': [1] * 20,
        'low': [1] * 20,
        'close': [1] * 20,
        'volume': [20] * 19 + [0.001],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=1,
        volume_threshold_ratio=0.5,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "min volume" in reason


def test_allow_trade_rejects_tiny_volume():
    data = {
        "open": [1] * 20,
        "high": [1] * 20,
        "low": [1] * 20,
        "close": [1] * 20,
        "volume": [50] * 19 + [0.1],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=5,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "min volume" in reason


def test_allow_trade_rejects_when_volume_far_below_mean():
    data = {
        "open": [1] * 20,
        "high": [1] * 20,
        "low": [1] * 20,
        "close": [1] * 20,
        "volume": [100] * 19 + [0.2],
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_volume=10,
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "min volume" in reason


def test_allow_trade_rejects_on_volatility_drop():
    data = {
        "open": [1] * 21,
        "high": [1] * 21,
        "low": [1] * 21,
        "close": [100] + [1] * 20,
        "volume": [10] * 21,
    }
    df = pd.DataFrame(data)
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "volatility" in reason.lower()


def test_stop_order_management():
    manager = RiskManager(RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01))
    order = {"id": "1", "symbol": "XBT/USDT", "side": "sell", "amount": 1, "dry_run": True}
    manager.register_stop_order(order, symbol="XBT/USDT")
    assert manager.stop_orders["XBT/USDT"]["amount"] == 1
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


def test_position_size_uses_trade_size_pct_when_no_stop():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            trade_size_pct=0.2,
        )
    )
    size = manager.position_size(0.5, 1000)
    assert size == 1000 * 0.2 * 0.5


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
    cfg = RiskConfig(max_drawdown=0.5, stop_loss_pct=0.01, take_profit_pct=0.01)
    manager = RiskManager(cfg)
    df = make_vol_df(1.0, 2.0)
    manager.peak_equity = 1.0
    manager.equity = 0.75
    short_atr = calc_atr(df, window=cfg.atr_short_window)
    long_atr = calc_atr(df, window=cfg.atr_long_window)
    vol_factor = min(short_atr / long_atr, cfg.max_volatility_factor)
    drawdown = 1 - manager.equity / manager.peak_equity
    risk_factor = 1 - drawdown / cfg.max_drawdown
    expected = 1000 * cfg.trade_size_pct * vol_factor * risk_factor
    size = manager.position_size(1.0, 1000, df)
    assert abs(size - expected) < 1e-6


def test_position_size_reduces_when_drawdown_high():
    cfg = RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01)
    manager = RiskManager(cfg)
    manager.peak_equity = 1.0
    manager.equity = 0.5
    df = make_vol_df(1.0, 1.0)
    size = manager.position_size(1.0, 1000, df)
    assert size == 1000 * cfg.trade_size_pct * 0.5
def test_position_size_risk_based():
    manager = RiskManager(
        RiskConfig(
            max_drawdown=1,
            stop_loss_pct=0.01,
            take_profit_pct=0.01,
            risk_pct=0.02,
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
        )
    )
    size = manager.position_size(0.5, 1000, stop_distance=0.02, price=0.1)
    assert abs(size - 50.0) < 1e-6


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
    assert cfg.volume_threshold_ratio == 0.1

def test_allow_trade_blocks_when_memory_hits_threshold(tmp_path, monkeypatch):
    data = {
        'open': [1] * 20,
        'high': [1] * 20,
        'low': [1] * 20,
        'close': [1] * 20,
        'volume': [10] * 20,
    }
    df = pd.DataFrame(data)
    mem = tmp_path / "mem.json"
    monkeypatch.setattr(trade_memory, "LOG_FILE", mem)
    trade_memory.configure(max_losses=1, slippage_threshold=0.5, lookback_seconds=3600)
    trade_memory.clear()
    trade_memory.record_loss("XBT/USDT", 0.01)

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        symbol="XBT/USDT",
    )
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "trade memory" in reason


def _df() -> pd.DataFrame:
    data = {
        "open": [1] * 20,
        "high": [1] * 20,
        "low": [1] * 20,
        "close": [1] * 20,
        "volume": [10] * 20,
    }
    return pd.DataFrame(data)


def test_allow_trade_rejects_on_negative_ev(tmp_path, monkeypatch):
    stats = {"trend_bot": {"win_rate": 0.4, "avg_win": 0.01, "avg_loss": -0.02}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))
    monkeypatch.setattr(ev_tracker, "STATS_FILE", file)

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.0,
    )
    allowed, reason = RiskManager(cfg).allow_trade(_df(), "trend_bot")
    assert not allowed
    assert "expected value" in reason.lower()


def test_allow_trade_allows_on_positive_ev(tmp_path, monkeypatch):
    stats = {"trend_bot": {"win_rate": 0.7, "avg_win": 0.02, "avg_loss": -0.01}}
    file = tmp_path / "stats.json"
    file.write_text(json.dumps(stats))
    monkeypatch.setattr(ev_tracker, "STATS_FILE", file)

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.0,
    )
    allowed, _ = RiskManager(cfg).allow_trade(_df(), "trend_bot")
    assert allowed


def test_allow_trade_ignores_ev_when_stats_missing(tmp_path, monkeypatch):
    file = tmp_path / "stats.json"
    file.write_text("{}")
    monkeypatch.setattr(ev_tracker, "STATS_FILE", file)

    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        min_expected_value=0.5,
    )
    allowed, _ = RiskManager(cfg).allow_trade(_df(), "trend_bot")
    assert allowed


def test_ev_tracker_logs_missing_file_once(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(ev_tracker, "STATS_FILE", missing)
    # reset warning flag in case other tests altered it
    monkeypatch.setattr(ev_tracker, "_missing_warning_emitted", False, raising=False)

    with caplog.at_level(logging.WARNING):
        ev_tracker.get_expected_value("trend_bot")
        ev_tracker.get_expected_value("grid_bot")

    messages = [r.getMessage() for r in caplog.records if "Strategy stats file" in r.getMessage()]
    assert len(messages) == 1


def test_allow_trade_rejects_on_pair_drawdown():
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
    allowed, reason = RiskManager(cfg).allow_trade(df)
    assert not allowed
    assert "drawdown" in reason.lower()


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
    allowed, _ = RiskManager(cfg).allow_trade(df)
    assert allowed
