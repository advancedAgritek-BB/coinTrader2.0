from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_drawdown: float
    stop_loss_pct: float
    take_profit_pct: float


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.equity = 1.0
        self.peak_equity = 1.0

    def update_equity(self, new_equity: float) -> bool:
        self.equity = new_equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        drawdown = 1 - self.equity / self.peak_equity
        return drawdown < self.config.max_drawdown

    def position_size(self, confidence: float, balance: float) -> float:
        return balance * confidence * 0.1

    def allow_trade(self, df) -> bool:
        """Return False when volume is low or volatility is flat."""
        if len(df) < 20:
            return False
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if df['volume'].iloc[-1] < vol_mean * 0.5:
            return False
        vol_std = df['close'].rolling(20).std().iloc[-1]
        if vol_std < df['close'].iloc[-20:-1].std() * 0.5:
            return False
        return True
