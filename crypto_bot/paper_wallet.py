class PaperWallet:
    """Simple wallet for paper trading supporting multiple positions."""

    def __init__(self, balance: float, max_open_trades: int = 1) -> None:
        self.balance = balance
        self.positions: list[dict] = []
        self.realized_pnl = 0.0
        self.max_open_trades = max_open_trades

    @property
    def position_size(self) -> float:
        return sum(p["amount"] for p in self.positions)

    @property
    def entry_price(self) -> float | None:
        if not self.positions:
            return None
        total = sum(p["amount"] * p["price"] for p in self.positions)
        return total / self.position_size if self.position_size else None

    @property
    def side(self) -> str | None:
        if not self.positions:
            return None
        first = self.positions[0]["side"]
        if all(p["side"] == first for p in self.positions):
            return first
        return "mixed"

    def open(self, side: str, amount: float, price: float) -> None:
        if len(self.positions) >= self.max_open_trades:
            raise RuntimeError("Position limit reached")
        if side == "buy":
            cost = amount * price
            if cost > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= cost
        else:
            self.balance += amount * price
        self.positions.append({"side": side, "amount": amount, "price": price})

    def close(self, amount: float, price: float) -> float:
        if not self.positions:
            return 0.0
        remaining = amount
        pnl_total = 0.0
        while remaining > 0 and self.positions:
            pos = self.positions[0]
            qty = min(remaining, pos["amount"])
            if pos["side"] == "buy":
                self.balance += qty * price
                pnl = (price - pos["price"]) * qty
            else:
                self.balance -= qty * price
                pnl = (pos["price"] - price) * qty
            pos["amount"] -= qty
            remaining -= qty
            if pos["amount"] <= 0:
                self.positions.pop(0)
            pnl_total += pnl
        self.realized_pnl += pnl_total
        return pnl_total

    def unrealized(self, price: float) -> float:
        pnl = 0.0
        for pos in self.positions:
            if pos["side"] == "buy":
                pnl += (price - pos["price"]) * pos["amount"]
            else:
                pnl += (pos["price"] - price) * pos["amount"]
        return pnl
