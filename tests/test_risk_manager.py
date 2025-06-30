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
    assert manager.allow_trade(df) is False
