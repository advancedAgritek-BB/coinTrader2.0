import asyncio
import json
import logging
from typing import Optional

import websockets


class KrakenWebsocketClient:
    """Minimal Kraken WebSocket v2 client supporting public and private feeds."""

    PUBLIC_URL = "wss://ws.kraken.com/v2"
    PRIVATE_URL = "wss://ws-auth.kraken.com/v2"

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token
        self.public_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.private_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def connect(self) -> None:
        """Connect to public (and private if token provided) websockets."""
        self.logger.info("Connecting to Kraken WebSocket API")
        self.public_ws = await websockets.connect(self.PUBLIC_URL)
        if self.token:
            self.private_ws = await websockets.connect(self.PRIVATE_URL)

    async def subscribe_ohlc(self, symbol: str, interval: int = 1) -> None:
        if not self.public_ws:
            raise RuntimeError("WebSocket not connected")
        msg = {
            "method": "subscribe",
            "params": {"channel": "ohlc", "symbol": symbol, "interval": interval},
        }
        await self.public_ws.send(json.dumps(msg))

    async def subscribe_book(self, symbol: str, depth: int = 10) -> None:
        if not self.public_ws:
            raise RuntimeError("WebSocket not connected")
        msg = {
            "method": "subscribe",
            "params": {"channel": "book", "symbol": symbol, "depth": depth},
        }
        await self.public_ws.send(json.dumps(msg))

    async def subscribe_executions(self) -> None:
        if not self.token or not self.private_ws:
            raise RuntimeError("Private WebSocket not connected")
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "executions",
                "token": self.token,
                "snap_trades": True,
                "snap_orders": True,
            },
        }
        await self.private_ws.send(json.dumps(msg))

    async def recv(self) -> dict:
        """Receive next message from any active socket."""
        if self.public_ws is None and self.private_ws is None:
            raise RuntimeError("No WebSocket connected")
        ws = self.public_ws or self.private_ws
        assert ws is not None
        data = await ws.recv()
        return json.loads(data)

    async def ping(self) -> None:
        if self.public_ws:
            await self.public_ws.send(json.dumps({"event": "ping"}))
        if self.private_ws:
            await self.private_ws.send(json.dumps({"event": "ping"}))


