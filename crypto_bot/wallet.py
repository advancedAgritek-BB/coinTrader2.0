from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Position:
    qty: float
    avg_price: float


class Wallet:
    """Simple wallet tracking balance and positions.

    Parameters
    ----------
    initial_balance:
        Starting liquid balance for the wallet.
    dry_run:
        If ``True`` (default) no real exchange calls will be made.  The class
        only updates its internal state.
    """

    def __init__(self, initial_balance: float = 10_000, dry_run: bool = True) -> None:
        self.balance = float(initial_balance)
        self.dry_run = dry_run
        # mapping symbol -> Position
        self.positions: Dict[str, Position] = {}
        # realized PnL per symbol
        self._realized: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Trade operations
    # ------------------------------------------------------------------
    def buy(self, symbol: str, qty: float, price: float) -> None:
        """Buy ``qty`` of ``symbol`` at ``price``.

        Updates the cash balance and average entry price if the position
        already exists.
        """

        cost = qty * price
        if cost > self.balance:
            raise RuntimeError("Insufficient balance")
        self.balance -= cost

        pos = self.positions.get(symbol)
        if pos:
            total_cost = pos.avg_price * pos.qty + cost
            new_qty = pos.qty + qty
            pos.avg_price = total_cost / new_qty
            pos.qty = new_qty
        else:
            self.positions[symbol] = Position(qty=qty, avg_price=price)
        if not self.dry_run:
            # Placeholder for real exchange interaction.
            pass

    def sell(self, symbol: str, qty: float, price: float) -> float:
        """Sell ``qty`` of ``symbol`` at ``price`` and return realized PnL."""

        pos = self.positions.get(symbol)
        if not pos or pos.qty < qty:
            raise RuntimeError("Position not sufficient to sell")

        proceeds = qty * price
        self.balance += proceeds

        realized = (price - pos.avg_price) * qty
        self._realized[symbol] = self._realized.get(symbol, 0.0) + realized

        pos.qty -= qty
        if pos.qty <= 0:
            del self.positions[symbol]
        if not self.dry_run:
            # Placeholder for real exchange interaction.
            pass
        return realized

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def update_pnl(self, symbol: str, current_price: float) -> float:
        """Return total PnL (realized + unrealized) for ``symbol``."""

        realized = self._realized.get(symbol, 0.0)
        pos = self.positions.get(symbol)
        if pos:
            unrealized = (current_price - pos.avg_price) * pos.qty
        else:
            unrealized = 0.0
        return realized + unrealized

    def total_balance(self, prices: Dict[str, float]) -> float:
        """Return liquid balance plus marked-to-market value of positions."""

        total = self.balance
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_price)
            total += pos.qty * price
        return total
