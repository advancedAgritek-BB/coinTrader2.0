"""Cross-chain arbitrage strategy comparing CEX and Solana prices."""

from __future__ import annotations

from typing import Iterable, Tuple

from crypto_bot.solana.api_helpers import fetch_solana_price


def generate_signal(
    exchanges: Iterable[object],
    symbol: str,
    threshold: float = 0.005,
) -> Tuple[float, str]:
    """Return a trading signal when CEX vs Solana spread exceeds ``threshold``.

    Parameters
    ----------
    exchanges : Iterable[object]
        Iterable of ccxt exchange instances providing ``fetch_ticker``.
    symbol : str
        Trading pair symbol like ``"SOL/USDC"``.
    threshold : float, optional
        Minimum fractional spread to trigger a signal. Defaults to ``0.005``.

    Returns
    -------
    tuple[float, str]
        ``(score, direction)`` where ``direction`` is ``"long"`` if the average
        CEX price is below the Solana price and ``"short"`` for the opposite
        case. ``score`` is the absolute spread capped at 1.0.
    """
    prices: list[float] = []
    for ex in exchanges:
        try:
            ticker = ex.fetch_ticker(symbol)
        except Exception:
            continue
        price = float(ticker.get("last") or 0)
        if price:
            prices.append(price)

    sol_price = fetch_solana_price(symbol)
    if not prices or sol_price <= 0:
        return 0.0, "none"

    cex_price = sum(prices) / len(prices)
    spread = (cex_price - sol_price) / sol_price
    score = abs(spread)

    if spread > threshold:
        return min(score, 1.0), "short"
    if spread < -threshold:
        return min(score, 1.0), "long"
    return 0.0, "none"
