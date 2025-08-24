import asyncio
import logging
from collections import defaultdict
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

async def build_tradable_set(
    exchange,
    *,
    allowed_quotes: Sequence[str] | None,
    min_daily_volume_quote: float,
    max_spread_pct: float,
    status: str = "online",
    whitelist: Iterable[str] | None = None,
    blacklist: Iterable[str] | None = None,
    max_pairs: int | None = None,
    max_pairs_total: int | None = None,
) -> list[str]:
    """Return symbols meeting basic liquidity/spread criteria.

    Markets and tickers are fetched once from *exchange* and filtered by the
    provided constraints.  If ``max_pairs`` is set, only the top-N pairs by
    quote volume are kept for each quote currency.  After this per-quote
    selection, the combined list is truncated to ``max_pairs_total`` if
    provided.
    """

    allowed_quotes_set = {str(q).upper() for q in (allowed_quotes or [])}
    whitelist_set = set(whitelist or [])
    blacklist_set = {s.upper() for s in blacklist or []}

    markets = exchange.list_markets()
    if asyncio.iscoroutine(markets):
        markets = await markets
    if isinstance(markets, list):
        markets = {m: {} for m in markets}

    tickers = exchange.fetch_tickers()
    if asyncio.iscoroutine(tickers):
        tickers = await tickers

    discovered = len(markets)
    tradable: list[tuple[str, str, float]] = []
    for symbol, info in markets.items():
        if whitelist_set and symbol not in whitelist_set:
            continue
        if symbol.upper() in blacklist_set:
            continue
        m_status = info.get("status")
        if status and m_status and m_status != status:
            continue
        base, _, quote = symbol.partition("/")
        quote = (info.get("quote") or quote).upper()
        if allowed_quotes_set and quote not in allowed_quotes_set:
            continue
        ticker = tickers.get(symbol, {}) if isinstance(tickers, dict) else {}
        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        if not bid or not ask:
            continue
        spread_pct = (ask - bid) / ask * 100
        if spread_pct > max_spread_pct:
            continue
        volume = float(
            ticker.get("quoteVolume")
            or ticker.get("baseVolume")
            or ticker.get("info", {}).get("quoteVolume", 0)
            or 0
        )
        if volume < min_daily_volume_quote:
            continue
        tradable.append((symbol, quote, volume))

    if max_pairs is not None:
        per_quote: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sym, quote, vol in tradable:
            per_quote[quote].append((sym, vol))
        selected: list[str] = []
        for quote, items in per_quote.items():
            items.sort(key=lambda x: x[1], reverse=True)
            selected.extend([sym for sym, _ in items[:max_pairs]])
    else:
        selected = [sym for sym, _, _ in tradable]

    if max_pairs_total is not None:
        selected = selected[:max_pairs_total]

    logger.info(
        "Discovered=%d \u2192 tradable=%d (quotes: %s, max_pairs=%s, max_pairs_total=%s)",
        discovered,
        len(selected),
        sorted(allowed_quotes_set) if allowed_quotes_set else sorted({q for _, q, _ in tradable}),
        max_pairs,
        max_pairs_total,
    )
    if max_pairs_total and len(selected) < max_pairs_total:
        logger.warning(
            "Filtered symbols fewer than requested max_pairs_total=%s (got %d)",
            max_pairs_total,
            len(selected),
        )
    logger.debug("Final tradable symbols: %s", ", ".join(sorted(selected)))
    return selected
