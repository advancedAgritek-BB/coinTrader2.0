import pandas as pd
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig

def test_allow_trade_rejects_low_volume():
    data = {
        'open':[1]*20,
        'high':[1]*20,
        'low':[1]*20,
        'close':[1]*20,
        'volume':[2]*19 + [0]
    }
    df = pd.DataFrame(data)
    manager = RiskManager(RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01))
    allowed, reason = manager.allow_trade(df)
    assert not allowed
    assert "Volume" in reason


def test_stop_order_management():
    manager = RiskManager(RiskConfig(max_drawdown=1, stop_loss_pct=0.01, take_profit_pct=0.01))
    order = {"id": "1", "symbol": "BTC/USDT", "side": "sell", "amount": 1, "dry_run": True}
    manager.register_stop_order(order)
    assert manager.stop_order["amount"] == 1
    manager.update_stop_order(0.5)
    assert manager.stop_order["amount"] == 0.5

    class DummyEx:
        def __init__(self):
            self.cancelled = False

        def cancel_order(self, oid, symbol):
            self.cancelled = True

    ex = DummyEx()
    manager.cancel_stop_order(ex)
    assert manager.stop_order is None


def test_position_size_uses_trade_size_pct():
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


def test_can_allocate_uses_tracker():
    cfg = RiskConfig(
        max_drawdown=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        strategy_allocation={"trend_bot": 0.5},
    )
    manager = RiskManager(cfg)
    assert manager.can_allocate("trend_bot", 40, 100)
    manager.allocate_capital("trend_bot", 40)
    assert not manager.can_allocate("trend_bot", 20, 100)
