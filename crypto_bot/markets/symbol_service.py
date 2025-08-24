import logging
from functools import lru_cache

import requests

try:  # pragma: no cover - the global cfg may not exist in tests
    from crypto_bot.config import cfg  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    from types import SimpleNamespace

    cfg = SimpleNamespace(strict_cex=False, denylist_symbols=[], allowed_quotes=[], min_volume=0.0)  # type: ignore


class SymbolService:
    """Utility service for generating candidate trading pairs.

    The service operates in "CEX" mode when working with a centralised exchange
    object that exposes ``list_markets`` and optional ``markets`` attributes.
    """

    def __init__(self, exchange):
        self.exchange = exchange
        # local logger per instance for consistency with other services
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    def _quote_ok(self, symbol: str) -> bool:
        allowed = {str(q).upper() for q in getattr(cfg, "allowed_quotes", []) or []}
        if not allowed:
            return True
        quote = str(symbol).split("/")[-1].upper()
        return quote in allowed

    def _volume_ok(self, symbol: str) -> bool:  # pragma: no cover - simple heuristic
        markets = getattr(self.exchange, "markets", {}) or {}
        info = markets.get(symbol, {}) if isinstance(markets, dict) else {}
        vol = float(info.get("quoteVolume") or info.get("baseVolume") or 0)
        min_vol = float(getattr(cfg, "min_volume", 0.0) or 0.0)
        return vol >= min_vol

    @staticmethod
    @lru_cache(maxsize=1)
    def _kraken_symbols() -> set[str]:  # pragma: no cover - network best effort
        """Return Kraken spot symbols from the public AssetPairs endpoint."""
        url = "https://api.kraken.com/0/public/AssetPairs"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json().get("result", {})
            syms = set()
            for info in data.values():
                ws = info.get("wsname") or ""
                if ws:
                    syms.add(ws.replace(" ", "/"))
            return syms
        except Exception:  # pragma: no cover - external API
            logging.getLogger(__name__).warning(
                "Failed to fetch Kraken asset list", exc_info=True
            )
            return set()

    # ------------------------------------------------------------------
    def get_candidates(self) -> list[str]:
        """Return candidate symbols for CEX trading.

        When ``cfg.strict_cex`` is enabled the list of markets is sourced
        directly from the exchange and filtered based on quote and volume
        checks. A runtime denylist (``cfg.denylist_symbols``) is also applied to
        exclude problematic pairs such as synthetic indexes.
        """

        if getattr(cfg, "strict_cex", False):
            markets = self.exchange.list_markets()  # authoritative symbols
            allowed = set(
                m for m in markets if self._quote_ok(m) and self._volume_ok(m)
            )
            deny = set(getattr(cfg, "denylist_symbols", []) or [])
            before = len(allowed)
            allowed.difference_update(deny)
            for bad in sorted(deny):
                if bad not in markets:
                    self.logger.info(
                        "Denylisted symbol not in exchange markets (ok): %s", bad
                    )
                else:
                    self.logger.info(
                        "Purged denylisted symbol from candidates: %s", bad
                    )
            kraken = self._kraken_symbols()
            if kraken:
                before_kraken = len(allowed)
                allowed.intersection_update(kraken)
                self.logger.info(
                    "Kraken asset filter: %d → %d", before_kraken, len(allowed)
                )
            self.logger.info(
                "CEX candidates: %d → %d after denylist", before, len(allowed)
            )
            return sorted(allowed)

        # Non-strict mode falls back to whatever the exchange already knows
        markets = getattr(self.exchange, "markets", {})
        if isinstance(markets, dict):
            symbols = markets.keys()
        else:
            symbols = markets or []
        return sorted(
            m for m in symbols if self._quote_ok(m) and self._volume_ok(m)
        )
