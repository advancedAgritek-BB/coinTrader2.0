class PaperWallet:
    """Simple wallet for paper trading supporting multiple open positions."""

    def __init__(self, balance: float) -> None:
        self.balance = balance
        # Mapping of trade identifier to position details
        # {id: {"side": str, "amount": float, "entry_price": float}}
        self.positions: dict[str, dict[str, float]] = {}
        self.realized_pnl = 0.0

    def open(
        self,
        side: str,
        amount: float,
        price: float,
        identifier: str | None = None,
    ) -> str:
        """Open a new paper trade and return its identifier."""
        from uuid import uuid4

        trade_id = identifier or str(uuid4())
        if side == "buy":
            cost = amount * price
            if cost > self.balance:
                raise ValueError("Insufficient balance")
            self.balance -= cost
        else:
            self.balance += amount * price

        self.positions[trade_id] = {
            "side": side,
            "amount": amount,
            "entry_price": price,
        }
        return trade_id

    def close(
        self,
        amount: float,
        price: float,
        identifier: str | None = None,
    ) -> float:
        """Close an existing trade. Returns realized PnL."""

        if not self.positions:
            return 0.0

        if identifier is None:
            if len(self.positions) != 1:
                # ambiguous position selection
                return 0.0
            identifier = next(iter(self.positions))

        pos = self.positions.get(identifier)
        if not pos:
            return 0.0

        amount = min(amount, pos["amount"])

        if pos["side"] == "buy":
            self.balance += amount * price
            pnl = (price - pos["entry_price"]) * amount
        else:
            self.balance -= amount * price
            pnl = (pos["entry_price"] - price) * amount

        pos["amount"] -= amount
        if pos["amount"] <= 0:
            del self.positions[identifier]
        else:
            self.positions[identifier] = pos

        self.realized_pnl += pnl
        return pnl

    def unrealized(self, price: float | dict[str, float]) -> float:
        """Return unrealized PnL across all open positions."""

        if not self.positions:
            return 0.0

        if isinstance(price, dict):
            total = 0.0
            for pid, pos in self.positions.items():
                if pid not in price:
                    continue
                p = price[pid]
                if pos["side"] == "buy":
                    total += (p - pos["entry_price"]) * pos["amount"]
                else:
                    total += (pos["entry_price"] - p) * pos["amount"]
            return total

        total = 0.0
        for pos in self.positions.values():
            if pos["side"] == "buy":
                total += (price - pos["entry_price"]) * pos["amount"]
            else:
                total += (pos["entry_price"] - price) * pos["amount"]
        return total
