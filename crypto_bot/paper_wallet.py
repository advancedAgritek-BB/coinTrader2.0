class PaperWallet:
    """Simple wallet for paper trading supporting multiple positions."""

    def __init__(self, balance: float) -> None:
        self.balance = balance
        self.positions: dict[str, dict] = {}
        self.realized_pnl = 0.0

    def open(self, symbol: str, side: str, amount: float, price: float) -> None:
        if symbol in self.positions:
            raise RuntimeError("Position already open for symbol")
        if side == "buy":
            cost = amount * price
            if cost > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= cost
        else:
            self.balance += amount * price
        self.positions[symbol] = {
            "side": side,
            "size": amount,
            "entry_price": price,
        }

    def close(self, symbol: str, amount: float, price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        amount = min(amount, pos["size"])
        if pos["side"] == "buy":
            self.balance += amount * price
            pnl = (price - pos["entry_price"]) * amount
        else:
            self.balance -= amount * price
            pnl = (pos["entry_price"] - price) * amount
        pos["size"] -= amount
        if pos["size"] <= 0:
            del self.positions[symbol]
        else:
            self.positions[symbol] = pos
        self.realized_pnl += pnl
        return pnl

    def unrealized(self, symbol: str, price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        if pos["side"] == "buy":
            return (price - pos["entry_price"]) * pos["size"]
        return (pos["entry_price"] - price) * pos["size"]
