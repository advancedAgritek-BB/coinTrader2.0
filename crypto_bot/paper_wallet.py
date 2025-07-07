class PaperWallet:
    """Simple wallet for paper trading supporting multiple positions."""

    def __init__(self, balance: float, max_open_trades: int = 1) -> None:
        self.balance = balance
        self.positions: list[dict] = []
        self.positions: dict[str, dict] = {}
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
    def open(self, symbol: str, side: str, amount: float, price: float) -> None:
        if symbol in self.positions:
            raise RuntimeError("Position already open for symbol")
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
