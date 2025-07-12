"""Cross-exchange arbitrage signal generation."""

from typing import Tuple


def generate_signal(exchange_a, exchange_b, symbol: str, threshold: float = 0.005) -> Tuple[float, str]:
    """Return a trading signal when price discrepancy exceeds ``threshold``.

    Parameters
    ----------
    exchange_a, exchange_b
        ccxt exchange instances used to fetch tickers.
    symbol : str
        Trading pair symbol (e.g. ``"BTC/USDT"``).
    threshold : float, optional
        Minimum fractional spread to trigger a signal. Defaults to ``0.005`` (0.5%).

    Returns
    -------
    tuple[float, str]
        ``(score, direction)`` where ``direction`` is ``"long"`` if ``exchange_a``
        price is below ``exchange_b`` and ``"short"`` for the opposite case.
    """
    try:
        ticker_a = exchange_a.fetch_ticker(symbol)
        ticker_b = exchange_b.fetch_ticker(symbol)
    except Exception:
        return 0.0, "none"

    price_a = float(ticker_a.get("last") or 0)
    price_b = float(ticker_b.get("last") or 0)
    if not price_a or not price_b:
        return 0.0, "none"

    spread = (price_a - price_b) / price_b
    score = abs(spread)

    if spread > threshold:
        return min(score, 1.0), "short"
    if spread < -threshold:
        return min(score, 1.0), "long"
    return 0.0, "none"
