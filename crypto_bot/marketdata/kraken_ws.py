"""Async Kraken market data websocket client.

Provides lightweight order book and trade stream handling with
computed microstructure metrics for use in high frequency trading
strategies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

PUBLIC_URL = "wss://ws.kraken.com/v2"


@dataclass
class Snapshot:
    """Lightweight market snapshot."""

    ts: float
    symbol: str
    mid: float
    spread_bp: float
    microprice: float
    obi: float
    depth_skew: float
    trade_imbalance_ewm: float
    rv_short: float


class KrakenWS:
    """Kraken public websocket client producing order book snapshots."""

    def __init__(self, symbols: List[str], exchange, depth: int = 10):
        self.symbols = symbols
        self.exchange = exchange
        self.depth = depth
        self.pairs: Dict[str, str] = {}
        self.symbol_from_pair: Dict[str, str] = {}
        self._map_symbols()

        self.books: Dict[str, Dict[str, Dict[float, float]]] = {
            sym: {
                "bids": {},
                "asks": {},
                "sequence": 0,
                "trade_ewm": 0.0,
                "last_trade_ts": None,
                "mid_history": deque(),
            }
            for sym in symbols
        }
        self.queue: asyncio.Queue[Snapshot] = asyncio.Queue(maxsize=1000)
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.ewm_window = 5.0  # seconds
        self.rv_window = 5.0  # seconds

    # ------------------------------------------------------------------
    def _map_symbols(self) -> None:
        self.exchange.load_markets()
        for sym in self.symbols:
            pair = self.exchange.market(sym)["id"]
            self.pairs[sym] = pair
            self.symbol_from_pair[pair] = sym

    # ------------------------------------------------------------------
    async def _subscribe(self, symbols: List[str]) -> None:
        if not self.ws:
            return
        await self.ws.send_json(
            {
                "method": "subscribe",
                "params": {
                    "channel": "book",
                    "symbol": symbols,
                    "depth": self.depth,
                    "snapshot": True,
                },
            }
        )
        await self.ws.send_json(
            {"method": "subscribe", "params": {"channel": "trade", "symbol": symbols}}
        )

    async def _resubscribe(self, sym: str) -> None:
        pair = self.pairs[sym]
        logger.warning("Resubscribing %s due to sequence gap", sym)
        await self._subscribe([pair])

    # ------------------------------------------------------------------
    async def _handle_book_snapshot(self, sym: str, data: Dict[str, List[List[str]]]) -> None:
        book = self.books[sym]
        book["bids"].clear()
        book["asks"].clear()
        seq = 0
        for side in ("bids", "asks"):
            for price, size, _ts, s in data.get(side, []):
                p = float(price)
                v = float(size)
                if v > 0:
                    book[side][p] = v
                seq = max(seq, int(s))
        book["sequence"] = seq

    async def _handle_book_update(self, sym: str, data: Dict[str, List[List[str]]]) -> None:
        book = self.books[sym]
        for side in ("bids", "asks"):
            for price, size, _ts, seq_str in data.get(side, []):
                seq = int(seq_str)
                if book["sequence"] and seq != book["sequence"] + 1:
                    await self._resubscribe(sym)
                    return
                book["sequence"] = seq
                p = float(price)
                v = float(size)
                levels = book[side]
                if v == 0:
                    levels.pop(p, None)
                else:
                    levels[p] = v

    def _update_trade_imbalance(self, sym: str, volume: float, ts: float) -> None:
        book = self.books[sym]
        last_ts = book["last_trade_ts"]
        if last_ts is None:
            book["trade_ewm"] = volume
        else:
            dt = ts - last_ts
            decay = math.exp(-dt / self.ewm_window)
            book["trade_ewm"] = decay * book["trade_ewm"] + (1 - decay) * volume
        book["last_trade_ts"] = ts

    async def _handle_trade(self, sym: str, trades: List[List[str]]) -> None:
        for price, size, ts, side, *_rest in trades:
            vol = float(size)
            sign = 1.0 if side == "buy" else -1.0
            self._update_trade_imbalance(sym, sign * vol, float(ts))

    # ------------------------------------------------------------------
    async def _process_message(self, msg: Dict) -> None:
        channel = msg.get("channel")
        sym = self.symbol_from_pair.get(msg.get("symbol", ""))
        if channel == "book" and sym:
            data = msg.get("data", {})
            if msg.get("type") == "snapshot":
                await self._handle_book_snapshot(sym, data)
            elif msg.get("type") == "update":
                await self._handle_book_update(sym, data)
        elif channel == "trade" and sym:
            await self._handle_trade(sym, msg.get("data", []))

    # ------------------------------------------------------------------
    def _make_snapshot(self, sym: str) -> Optional[Snapshot]:
        book = self.books[sym]
        bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[: self.depth]
        asks = sorted(book["asks"].items(), key=lambda x: x[0])[: self.depth]
        if not bids or not asks:
            return None
        best_bid, bid_sz = bids[0]
        best_ask, ask_sz = asks[0]
        mid = (best_bid + best_ask) / 2
        spread_bp = (best_ask - best_bid) / mid * 1e4
        bid_vol = sum(v for _p, v in bids)
        ask_vol = sum(v for _p, v in asks)
        microprice = (best_ask * bid_sz + best_bid * ask_sz) / (bid_sz + ask_sz)
        denom = bid_vol + ask_vol
        obi = (bid_vol - ask_vol) / denom if denom else 0.0
        depth_skew = bid_vol / ask_vol if ask_vol else float("inf")

        ts = time.time()
        hist: deque = book["mid_history"]
        hist.append((ts, mid))
        while hist and ts - hist[0][0] > self.rv_window:
            hist.popleft()
        rets = [math.log(hist[i][1] / hist[i - 1][1]) for i in range(1, len(hist))]
        rv = sum(r * r for r in rets)

        return Snapshot(
            ts=ts,
            symbol=sym,
            mid=mid,
            spread_bp=spread_bp,
            microprice=microprice,
            obi=obi,
            depth_skew=depth_skew,
            trade_imbalance_ewm=book["trade_ewm"],
            rv_short=rv,
        )

    async def _snapshot_loop(self, sym: str) -> None:
        while True:
            await asyncio.sleep(0.1)
            snap = self._make_snapshot(sym)
            if snap is None:
                continue
            try:
                self.queue.put_nowait(snap)
            except asyncio.QueueFull:
                logger.debug("Dropping snapshot for %s", sym)

    async def _connect_loop(self) -> None:
        backoff = 1
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    self.session = session
                    async with session.ws_connect(PUBLIC_URL, heartbeat=20) as ws:
                        self.ws = ws
                        logger.info("Connected to Kraken public websocket")
                        await self._subscribe(list(self.pairs.values()))
                        backoff = 1
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await self._process_message(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as exc:  # pragma: no cover - network error
                logger.warning("Websocket connection error: %s", exc)
            finally:
                self.ws = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def stream(self) -> AsyncIterator[Snapshot]:
        tasks = [asyncio.create_task(self._snapshot_loop(sym)) for sym in self.symbols]
        producer = asyncio.create_task(self._connect_loop())
        try:
            while True:
                snap = await self.queue.get()
                yield snap
        finally:
            producer.cancel()
            for t in tasks:
                t.cancel()
            if self.session:
                await self.session.close()


async def stream_snapshots(symbols: List[str], exchange) -> AsyncIterator[Snapshot]:
    """Yield Kraken order book snapshots for ``symbols``.

    The exchange argument must be an instantiated CCXT exchange
    object (synchronous is acceptable). The coroutine yields
    :class:`Snapshot` objects roughly every 100ms per symbol.
    """

    client = KrakenWS(symbols, exchange)
    async for snap in client.stream():
        yield snap
