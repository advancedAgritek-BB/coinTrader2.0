from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger

try:  # pragma: no cover - optional dependency
    from crypto_bot.solana.pump_fun_client import PumpFunClient
except Exception:  # pragma: no cover - fallback stub
    class PumpFunClient:  # type: ignore[override]
        async def trending(self) -> list[dict]:
            return []

        async def aclose(self) -> None:
            pass


try:  # pragma: no cover - optional dependency
    from crypto_bot.solana.sniper.selector import TokenSelector, TokenScore
except Exception:  # pragma: no cover - fallback stubs
    class TokenScore:  # minimal placeholder
        def __init__(
            self,
            mint: str = "",
            symbol: str | None = None,
            liquidity_usd: float = 0.0,
            volume24h_usd: float = 0.0,
            score: float = 0.0,
        ) -> None:
            self.mint = mint
            self.symbol = symbol
            self.liquidity_usd = liquidity_usd
            self.volume24h_usd = volume24h_usd
            self.score = score

class TokenSelector:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

        def score_mint(self, mint: str, created_unix: int | None = None) -> Optional[TokenScore]:
            return None

        def close(self) -> None:
            pass


class Signal:
    def __init__(self, symbol: str, side: str, qty: float, reason: str) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.reason = reason


class Strategy:  # <-- the loader will find this
    name = "sniper_solana"
    timeframes = ["1m"]

    def __init__(
        self,
        *,
        base_quote_mint: str = "So11111111111111111111111111111111111111112",  # SOL
        buy_notional_usd: float = 50.0,
        max_open_positions: int = 3,
        **_: Any,
    ) -> None:
        self.base_quote_mint = base_quote_mint
        self.buy_notional_usd = buy_notional_usd
        self.max_open_positions = max_open_positions
        self._pump = PumpFunClient()
        self._selector = TokenSelector(
            min_liquidity_usd=10_000.0,
            min_volume24h_usd=15_000.0,
            prefer_age_minutes=30.0,
            hard_min_age_sec=90,
        )

    async def aclose(self) -> None:
        await self._pump.aclose()
        self._selector.close()

    # ---- Integration contract (lightweight & forgiving) ---------------------
    # The evaluator should call one of these periodically with context:
    async def evaluate(self, context: Any) -> list[Signal]:
        """
        Pull trending/new tokens from Pump.fun, filter via Helius+Raydium,
        and return buy signals.
        """
        try:
            trending = await self._pump.trending()
        except Exception as e:
            logger.warning(f"pump.fun trending fetch failed: {e}")
            return []

        scored: list[TokenScore] = []
        for it in trending[:30]:  # cap requests
            mint = (it.get("mint") or it.get("address") or "").strip()
            if not mint:
                continue
            created_unix = it.get("createdTimestamp") or it.get("created_at") or it.get("timestamp")
            try:
                ts = int(created_unix) if created_unix is not None else None
            except Exception:
                ts = None
            try:
                sc = self._selector.score_mint(mint, created_unix=ts)
                if sc:
                    scored.append(sc)
            except Exception as e:
                logger.debug(f"score_mint error for {mint}: {e}")

        scored.sort(key=lambda s: s.score, reverse=True)
        take = scored[: self.max_open_positions]

        signals: list[Signal] = []
        for s in take:
            # Place a BUY signal using USD notional; executor will translate to qty.
            sym = f"{s.symbol or s.mint}/SOL"
            reason = f"score={s.score:.2f}; liq={s.liquidity_usd:.0f}; vol24h={s.volume24h_usd:.0f}"
            signals.append(Signal(symbol=sym, side="buy", qty=self.buy_notional_usd, reason=reason))
            logger.info(f"[sniper] BUY {sym} ({s.mint}) -> {reason}")

        return signals


# ---------------------------------------------------------------------------
# Legacy API used by some tests: lightweight ATR-like spike detector

def get_pyth_price(symbol: str, cfg: Optional[dict] = None) -> float:
    """Placeholder Pyth price fetcher (patched in tests)."""
    return 0.0


def generate_signal(df, config: Optional[dict] = None) -> tuple[float, str]:
    """Simplified jump-based signal used for backward compatibility."""
    if df is None or len(df) < 2:
        return 0.0, "none"

    params = config or {}
    if not bool(params.get("is_trading", True)) or float(params.get("conf_pct", 0.0)) > 0.5:
        return 0.0, "none"

    token = params.get("token")
    jump_mult = float(params.get("jump_mult", 4.0))

    df = df.copy()
    if token:
        try:
            df.loc[df.index[-1], "close"] = float(get_pyth_price(f"Crypto.{token}/USD", config))
        except Exception:
            pass

    prev_close = float(df["close"].iloc[-2])
    curr_close = float(df["close"].iloc[-1])
    price_change = curr_close - prev_close

    if abs(price_change) >= jump_mult:
        return 1.0, ("long" if price_change > 0 else "short")
    return 0.0, "none"
