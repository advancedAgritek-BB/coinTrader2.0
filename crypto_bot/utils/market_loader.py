from typing import Iterable, List


def load_kraken_symbols(exchange, exclude: Iterable[str] | None = None) -> List[str]:
    """Return a list of active trading pairs on Kraken.

    Parameters
    ----------
    exchange : ccxt Exchange
        Exchange instance connected to Kraken.
    exclude : Iterable[str] | None
        Symbols to exclude from the result.
    """
    exclude_set = set(exclude or [])
    markets = exchange.load_markets()
    symbols: List[str] = []
    for symbol, data in markets.items():
        if not data.get("active", True):
            continue
        if symbol in exclude_set:
            continue
        symbols.append(symbol)
    return symbols
