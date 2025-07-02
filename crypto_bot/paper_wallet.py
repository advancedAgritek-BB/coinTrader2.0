class PaperWallet:
    """Simple wallet for paper trading."""

    def __init__(self, balance: float) -> None:
        self.balance = balance
        self.position_size = 0.0
        self.entry_price: float | None = None
        self.side: str | None = None
        self.realized_pnl = 0.0

    def open(self, side: str, amount: float, price: float) -> None:
        if side == "buy":
            cost = amount * price
            if cost > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= cost
        else:
            self.balance += amount * price
        self.position_size = amount
        self.entry_price = price
        self.side = side

    def close(self, amount: float, price: float) -> float:
        if self.position_size == 0:
            return 0.0
        if self.side == "buy":
            self.balance += amount * price
            pnl = (price - (self.entry_price or 0)) * amount
        else:
            self.balance -= amount * price
            pnl = ((self.entry_price or 0) - price) * amount
        self.position_size -= amount
        if self.position_size <= 0:
            self.position_size = 0.0
            self.entry_price = None
            self.side = None
        self.realized_pnl += pnl
        return pnl

    def unrealized(self, price: float) -> float:
        if self.position_size and self.entry_price is not None and self.side:
            if self.side == "buy":
                return (price - self.entry_price) * self.position_size
            return (self.entry_price - price) * self.position_size
        return 0.0
