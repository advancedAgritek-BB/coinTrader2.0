import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable

from crypto_bot.utils.eval_guard import eval_gate

logger = logging.getLogger(__name__)


def _has_ohlcv(ctx: Any, symbol: str, timeframe: str, warmup_met: bool = True) -> bool:
    """Return True if OHLCV data for *symbol*/*timeframe* exists and meets warmup."""
    cache = getattr(ctx, "df_cache", {}) or {}
    tf_cache = cache.get(timeframe, {}) or {}
    df = tf_cache.get(symbol)
    if df is None or getattr(df, "empty", True):
        return False
    if warmup_met:
        warm_map = getattr(ctx, "config", {}).get("warmup_candles", {}) or {}
        required = int(warm_map.get(timeframe, 0) or 0)
        if required and len(df) < required:
            return False
    return True


class EvaluationEngine:
    """Evaluate strategies with gate protection and warmup checks."""

    def __init__(self, ttl: float = 120.0):
        self.ttl = ttl

    async def _watchdog(self, symbol: str, strat: str) -> None:
        try:
            await asyncio.sleep(self.ttl)
        except asyncio.CancelledError:  # pragma: no cover - watchdog cancelled
            return
        if eval_gate.is_busy():
            logger.warning(
                "Gate held >%ds by %s/%s; force-releasing",
                self.ttl,
                symbol,
                strat,
            )
            eval_gate._busy = False  # force release

    async def evaluate(
        self,
        symbol: str,
        strategy: Callable[[str, Any], Awaitable[Any]],
        ctx: Any,
    ) -> Any:
        """Evaluate *strategy* for *symbol* using *ctx* with gate management."""
        if not _has_ohlcv(ctx, symbol, "1m", warmup_met=True):
            logger.debug(
                "EVAL SKIP %s: prerequisites not met (missing 5m or warmup)", symbol
            )
            return None
        needs_5m = False
        try:
            lookback = getattr(strategy, "required_lookback")()
            needs_5m = "5m" in (lookback or {})
        except Exception:  # pragma: no cover - strategy missing hook
            needs_5m = False
        if needs_5m and not _has_ohlcv(ctx, symbol, "5m", warmup_met=True):
            logger.debug(
                "EVAL SKIP %s: prerequisites not met (missing 5m or warmup)", symbol
            )
            return None

        strat_name = getattr(strategy, "__name__", str(strategy))
        owner = threading.get_ident()
        logger.debug("Gate acquire by %s/%s (owner=%s)", symbol, strat_name, owner)
        watchdog = asyncio.create_task(self._watchdog(symbol, strat_name))
        try:
            with eval_gate.hold(f"{symbol}/{strat_name}"):
                return await strategy(symbol, ctx)
        finally:
            watchdog.cancel()
            logger.debug(
                "Gate release by %s/%s (owner=%s)", symbol, strat_name, owner
            )
