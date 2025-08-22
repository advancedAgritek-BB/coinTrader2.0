from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4


class PaperWallet:
    """Simple wallet for paper trading supporting multiple positions."""

    def __init__(
        self,
        balance: float,
        max_open_trades: int = 10,
        allow_short: bool = True,
        stake_usd: float | None = None,
        min_price: float = 0.0,
        min_notional: float = 0.0,
    ) -> None:
        self.balance = balance
        # mapping of identifier (symbol or trade id) -> position details
        # each position: {"symbol": str | None, "side": str, "amount": float, "entry_price": float}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl = 0.0
        self.max_open_trades = max_open_trades
        self.allow_short = allow_short
        self.stake_usd = stake_usd
        self.min_price = min_price
        self.min_notional = min_notional

    # ------------------------------------------------------------------
    def _check_limits(self, price: float, amount: float) -> None:
        if price < self.min_price:
            raise RuntimeError("Price below minimum limit")
        cost = price * amount
        if cost < self.min_notional:
            raise RuntimeError("Notional below minimum limit")
        if self.stake_usd is not None and cost > self.stake_usd:
            raise RuntimeError("Stake exceeds limit")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def position_size(self) -> float:
        """Total size across all open positions."""
        total = 0.0
        for pos in self.positions.values():
            if "size" in pos:
                total += pos["size"]
            else:
                total += pos["amount"]
        return total

    @property
    def entry_price(self) -> float | None:
        if not self.positions:
            return None
        total_amt = self.position_size
        if not total_amt:
            return None
        total = 0.0
        for pos in self.positions.values():
            qty = pos.get("size", pos.get("amount", 0.0))
            total += qty * pos["entry_price"]
        return total / total_amt

    @property
    def side(self) -> str | None:
        if not self.positions:
            return None
        first = next(iter(self.positions.values()))["side"]
        if all(p["side"] == first for p in self.positions.values()):
            return first
        return "mixed"

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------
    def open(self, *args) -> str:
        """Open a new trade and return its identifier.

        Supported signatures:
            open(side, amount, price, identifier=None)
            open(symbol, side, amount, price, identifier=None)
        """

        if not args:
            raise TypeError("open() missing required arguments")

        if args[0] in {"buy", "sell"}:
            side = args[0]
            amount = args[1]
            price = args[2]
            identifier = args[3] if len(args) > 3 else None
            symbol = None
        else:
            symbol = args[0]
            side = args[1]
            amount = args[2]
            price = args[3]
            identifier = args[4] if len(args) > 4 else None

            if symbol in self.positions:
                raise RuntimeError("Position already open for symbol")

        if side == "sell" and not self.allow_short:
            raise RuntimeError("Short selling disabled")

        if len(self.positions) >= self.max_open_trades:
            raise RuntimeError("Position limit reached")

        self._check_limits(price, amount)
        trade_id = identifier or symbol or str(uuid4())

        cost = amount * price
        reserved = 0.0
        if side == "buy":
            if cost > self.balance:
                raise RuntimeError("Insufficient balance")
            self.balance -= cost
        else:
            if cost > self.balance:
                raise RuntimeError("Insufficient balance")
            self.balance -= cost
            reserved = cost

        if symbol is not None:
            self.positions[trade_id] = {
                "symbol": symbol,
                "side": side,
                "size": amount,
                "entry_price": price,
                "reserved": reserved,
            }
        else:
            self.positions[trade_id] = {
                "symbol": None,
                "side": side,
                "amount": amount,
                "entry_price": price,
                "reserved": reserved,
            }
        return trade_id

    def close(self, *args) -> float:
        """Close an existing position and return realized PnL.

        Supported signatures:
            close(symbol, amount, price)
            close(amount, price, identifier=None)
        """

        if not self.positions:
            return 0.0

        identifier: str | None = None
        amount: float
        price: float

        if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], (int, float)) and isinstance(args[2], (int, float)):
            identifier = args[0]
            amount = float(args[1])
            price = float(args[2])
        elif len(args) >= 2 and all(isinstance(a, (int, float)) for a in args[:2]):
            amount = float(args[0])
            price = float(args[1])
            identifier = args[2] if len(args) > 2 else None
            if identifier is None and len(self.positions) == 1:
                identifier = next(iter(self.positions))
        else:
            raise TypeError("Invalid arguments for close()")

        if identifier is None:
            return 0.0

        pos = self.positions.get(identifier)
        if not pos:
            return 0.0

        if price == 0.0:
            price = pos["entry_price"]

        key = "size" if "size" in pos else "amount"
        amount = min(amount, pos[key])

        if "size" in pos:
            # symbol-based logic
            if pos["side"] == "buy":
                pnl = (price - pos["entry_price"]) * amount
                self.balance += amount * price
            else:
                pnl = (pos["entry_price"] - price) * amount
                release = pos["entry_price"] * amount
                self.balance += release + pnl
                pos["reserved"] -= release
            pos[key] -= amount
            self.realized_pnl += pnl
        else:
            if pos["side"] == "buy":
                pnl = (price - pos["entry_price"]) * amount
                self.balance += amount * price
            else:
                pnl = (pos["entry_price"] - price) * amount
                release = pos["entry_price"] * amount
                self.balance += release + pnl
                pos["reserved"] -= release
            pos[key] -= amount
            self.realized_pnl += pnl

        if pos[key] <= 0:
            del self.positions[identifier]
        else:
            self.positions[identifier] = pos

        return pnl

    def unrealized(self, *args) -> float:
        """Return unrealized PnL.

        Supported signatures:
            unrealized(price)
            unrealized(symbol, price)
            unrealized({id: price, ...})
        """

        if not self.positions:
            return 0.0

        if len(args) == 2 and isinstance(args[0], str):
            identifier = args[0]
            price = float(args[1])
            pos = self.positions.get(identifier)
            if not pos:
                return 0.0
            key = "size" if "size" in pos else "amount"
            if pos["side"] == "buy":
                return (price - pos["entry_price"]) * pos[key]
            return (pos["entry_price"] - price) * pos[key]

        if len(args) == 1:
            price = args[0]
            if isinstance(price, dict):
                total = 0.0
                for pid, p in price.items():
                    pos = self.positions.get(pid)
                    if not pos:
                        continue
                    key = "size" if "size" in pos else "amount"
                    if pos["side"] == "buy":
                        total += (p - pos["entry_price"]) * pos[key]
                    else:
                        total += (pos["entry_price"] - p) * pos[key]
                return total

            price_val = float(price)
            total = 0.0
            for pos in self.positions.values():
                key = "size" if "size" in pos else "amount"
                if pos["side"] == "buy":
                    total += (price_val - pos["entry_price"]) * pos[key]
                else:
                    total += (pos["entry_price"] - price_val) * pos[key]
            return total

        return 0.0

