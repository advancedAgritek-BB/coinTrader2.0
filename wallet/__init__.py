from __future__ import annotations

from typing import Any, Dict

from crypto_bot.paper_wallet import PaperWallet


class Wallet(PaperWallet):
    """Wallet used for dry-run trading.

    Extends :class:`~crypto_bot.paper_wallet.PaperWallet` with simple
    ``buy`` and ``sell`` helpers plus a ``total_balance`` property used by
    the rest of the bot when operating in dry-run mode.
    """

    @property
    def total_balance(self) -> float:
        """Return the current total balance.

        ``PaperWallet`` already maintains ``balance`` reflecting realized
        cash after accounting for open positions.  For dry-run purposes we
        expose it via ``total_balance`` to match the interface expected by
        the controller.
        """

        return self.balance

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------
    def buy(self, symbol: str, amount: float, price: float) -> None:
        """Execute a virtual buy order.

        If an existing short position is present it will be reduced using
        :meth:`PaperWallet.close`.  Otherwise a long position is opened or
        increased.
        """

        pos = self.positions.get(symbol)
        if pos and pos.get("side") == "sell":
            # buying back part of a short position
            self.close(symbol, amount, price)
            return
        self._check_limits(price, amount)

        if pos and pos.get("side") == "buy":
            # increase existing long without opening a new trade id
            cost = amount * price
            if cost > self.balance:
                raise RuntimeError("Insufficient balance")
            total = pos["size"] + amount
            pos["entry_price"] = ((pos["entry_price"] * pos["size"]) + (price * amount)) / total
            pos["size"] = total
            self.balance -= cost
            return

        # otherwise open a new long position
        self.open(symbol, "buy", amount, price)

    def sell(self, symbol: str, amount: float, price: float) -> None:
        """Execute a virtual sell order.

        If a long position exists it will be reduced via
        :meth:`PaperWallet.close`.  Otherwise a short position is opened or
        increased.
        """

        pos = self.positions.get(symbol)
        if pos and pos.get("side") == "buy":
            # selling part of a long position
            self.close(symbol, amount, price)
            return
        self._check_limits(price, amount)

        if pos and pos.get("side") == "sell":
            # increase existing short
            cost = amount * price
            if cost > self.balance:
                raise RuntimeError("Insufficient balance")
            total = pos["size"] + amount
            pos["entry_price"] = ((pos["entry_price"] * pos["size"]) + (price * amount)) / total
            pos["size"] = total
            pos["reserved"] += cost
            self.balance -= cost
            return

        # otherwise open a new short position
        self.open(symbol, "sell", amount, price)
