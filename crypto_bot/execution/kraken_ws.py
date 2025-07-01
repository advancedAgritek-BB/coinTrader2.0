import json
import threading
import os
from typing import Optional

import ccxt
from websocket import WebSocketApp

PUBLIC_URL = "wss://ws.kraken.com"
PRIVATE_URL = "wss://ws-auth.kraken.com"


class KrakenWSClient:
    """Minimal Kraken WebSocket client for public and private channels."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_secret = api_secret or os.getenv("API_SECRET")
        self.exchange = None
        if self.api_key and self.api_secret:
            self.exchange = ccxt.kraken(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                }
            )
        self.token: Optional[str] = None
        self.public_ws: Optional[WebSocketApp] = None
        self.private_ws: Optional[WebSocketApp] = None

    def get_token(self) -> str:
        """Retrieve WebSocket authentication token via Kraken REST API."""
        if not self.exchange:
            raise ValueError("API keys required for private websocket")
        resp = self.exchange.privatePostGetWebSocketsToken()
        self.token = resp["token"]
        return self.token

    def _start_ws(self, url: str, **kwargs) -> WebSocketApp:
        ws = WebSocketApp(url, **kwargs)
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        return ws

    def connect_public(self) -> None:
        if not self.public_ws:
            self.public_ws = self._start_ws(PUBLIC_URL)

    def connect_private(self) -> None:
        if not self.private_ws:
            if not self.token:
                self.get_token()
            self.private_ws = self._start_ws(PRIVATE_URL)

    def subscribe_ticker(self, pair: str) -> None:
        self.connect_public()
        msg = {"event": "subscribe", "pair": [pair], "subscription": {"name": "ticker"}}
        self.public_ws.send(json.dumps(msg))

    def subscribe_trades(self, pair: str) -> None:
        self.connect_public()
        msg = {"event": "subscribe", "pair": [pair], "subscription": {"name": "trade"}}
        self.public_ws.send(json.dumps(msg))

    def subscribe_orders(self) -> None:
        self.connect_private()
        msg = {"event": "subscribe", "subscription": {"name": "openOrders", "token": self.token}}
        self.private_ws.send(json.dumps(msg))

    def add_order(self, pair: str, side: str, volume: float, ordertype: str = "market") -> dict:
        self.connect_private()
        msg = {
            "event": "addOrder",
            "token": self.token,
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        self.private_ws.send(json.dumps(msg))
        return msg
