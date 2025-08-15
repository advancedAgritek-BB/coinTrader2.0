from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

# -- Optional dependencies -------------------------------------------------
# The real project provides dedicated clients for Helius and Raydium.  The
# execution environment used for the kata however does not ship these heavy
# dependencies.  Import errors would make this module impossible to import
# during testing, so we provide light‑weight fallbacks that can be monkey
# patched in the tests.
try:  # pragma: no cover - imported when available
    from crypto_bot.solana.helius_client import HeliusClient, TokenMetadata  # type: ignore
except Exception:  # pragma: no cover - simplified fallback
    @dataclass
    class TokenMetadata:  # minimal structure for tests
        mint_authority: Optional[str] = None
        freeze_authority: Optional[str] = None
        symbol: Optional[str] = None
        name: Optional[str] = None
        decimals: Optional[int] = None

    class HeliusClient:  # pragma: no cover - mocked in tests
        def get_token_metadata(self, _mint: str) -> Optional[TokenMetadata]:
            return None

        def close(self) -> None:  # pragma: no cover - nothing to cleanup
            pass

try:  # pragma: no cover - imported when available
    from crypto_bot.solana.raydium_client import RaydiumClient  # type: ignore
except Exception:  # pragma: no cover
    try:  # many parts of the project store the client under utils
        from crypto_bot.utils.raydium_client import RaydiumClient  # type: ignore
    except Exception:
        class RaydiumClient:  # pragma: no cover - mocked in tests
            def best_pool_for_mint(self, _mint: str, *, min_liquidity_usd: float = 0.0):
                return None

            def close(self) -> None:
                pass


@dataclass
class TokenScore:
    mint: str
    symbol: str
    name: str
    score: float
    reasons: list[str]
    pool_address: Optional[str]
    liquidity_usd: float
    volume24h_usd: float
    price: float


class TokenSelector:
    """Score and filter new Solana tokens for sniping.

    The selector evaluates freshly created tokens using a handful of heuristics
    that favour liquid, actively traded assets with sensible metadata.  The
    goal is to cheaply discard obviously bad or risky tokens before more
    expensive analysis takes place.
    """

    def __init__(
        self,
        *,
        min_liquidity_usd: float = 10_000.0,
        min_volume24h_usd: float = 15_000.0,
        prefer_age_minutes: float = 30.0,
        hard_min_age_sec: int = 90,
    ) -> None:
        self.min_liq = min_liquidity_usd
        self.min_vol = min_volume24h_usd
        self.prefer_age_min = prefer_age_minutes
        self.hard_min_age_sec = hard_min_age_sec
        self.hc = HeliusClient()
        self.rc = RaydiumClient()

    def close(self) -> None:
        self.rc.close()
        self.hc.close()

    # ------------------------------------------------------------------
    def _authority_ok(self, md: TokenMetadata) -> bool:
        """Return ``True`` if mint/freeze authorities are disabled.

        Tokens that still have an active mint or freeze authority are higher
        rug‑pull risks.  Users may choose to override this behaviour, but the
        selector treats such tokens as invalid.
        """

        if getattr(md, "freeze_authority", None) or getattr(md, "mint_authority", None):
            return False
        return True

    def _age_penalty(self, created_unix: Optional[int]) -> float:
        if not created_unix:
            return 0.0
        age_sec = max(1, int(time.time()) - int(created_unix))
        if age_sec < self.hard_min_age_sec:
            return -2.0  # too fresh
        age_min = age_sec / 60.0
        # sweet spot around prefer_age_min; within [5, 120] minutes is fine
        if 5 <= age_min <= 120:
            return 1.0 - abs(age_min - self.prefer_age_min) / self.prefer_age_min
        return -0.5

    # ------------------------------------------------------------------
    def score_mint(self, mint: str, created_unix: Optional[int] = None) -> Optional[TokenScore]:
        """Return :class:`TokenScore` for ``mint`` or ``None`` if rejected.

        The method performs a number of inexpensive checks.  If any hard rule
        fails the token is discarded outright (``None`` is returned).  Otherwise
        a numeric score is produced where higher is better.
        """

        md = self.hc.get_token_metadata(mint)
        if not md:
            return None

        reasons: list[str] = []
        if not self._authority_ok(md):
            reasons.append("authority_active")
            return None

        pool = self.rc.best_pool_for_mint(mint, min_liquidity_usd=self.min_liq)
        if not pool:
            reasons.append("no_pool")
            return None

        liq = float(pool.get("liquidityUsd") or 0.0)
        vol = float(pool.get("volume24hUsd") or 0.0)
        price = float(pool.get("price") or 0.0)

        if liq < self.min_liq:
            reasons.append(f"liquidity<{self.min_liq}")
            return None
        if vol < self.min_vol:
            reasons.append(f"vol24h<{self.min_vol}")
            return None

        sc = 0.0
        sc += min(2.0, math.log10(max(1.0, liq)) / 2.0)  # 0..~2
        sc += min(2.0, math.log10(max(1.0, vol)) / 2.0)  # 0..~2
        sc += self._age_penalty(created_unix)            # -2..1

        # ------------------------------------------------------------------
        # symbol/name sanity
        sym = (getattr(md, "symbol", None) or "").strip()
        name = (getattr(md, "name", None) or "").strip()
        if not sym:
            sc -= 0.5
            reasons.append("missing_symbol")
        if not name:
            sc -= 0.5
            reasons.append("missing_name")

        if sym:
            if len(sym) < 2 or len(sym) > 10 or not sym.isascii() or not sym.isupper() or not sym.isalnum():
                sc -= 0.2
                reasons.append("symbol_anomaly")

        # decimals sanity – most SPL tokens use 6 or 9 decimals.  Anything
        # outside 0‑9 or missing is suspicious and heavily penalised.
        decimals = getattr(md, "decimals", None)
        if decimals is None:
            sc -= 0.25
            reasons.append("decimals_missing")
        else:
            if decimals < 0 or decimals > 9:
                sc -= 1.0
                reasons.append("decimals_out_of_range")
            elif decimals not in (6, 9):
                sc -= 0.25
                reasons.append("unusual_decimals")

        return TokenScore(
            mint=mint,
            symbol=sym or mint[:4],
            name=name or mint[:6],
            score=sc,
            reasons=reasons,
            pool_address=pool.get("address"),
            liquidity_usd=liq,
            volume24h_usd=vol,
            price=price,
        )
