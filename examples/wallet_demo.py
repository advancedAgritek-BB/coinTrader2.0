from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from crypto_bot.paper_wallet import PaperWallet as Wallet


def main() -> None:
    """Demonstrate basic wallet operations."""
    # start with $1,000
    wallet = Wallet(balance=1000.0)

    # buy 1 unit at $100
    wallet.open("buy", 1, 100)

    # sell the unit at $110
    wallet.close(1, 110)

    print(f"Balance: ${wallet.balance:.2f}")
    print(f"Realized PnL: ${wallet.realized_pnl:.2f}")


if __name__ == "__main__":
    main()

