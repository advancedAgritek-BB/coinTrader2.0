import json
import threading
import os
from typing import Optional, Callable

import ccxt
from websocket import WebSocketApp
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/execution.log")

PUBLIC_URL = "wss://ws.kraken.com"
PRIVATE_URL = "wss://ws-auth.kraken.com"


class KrakenWSClient:
    """Minimal Kraken WebSocket client for public and private channels."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        ws_token: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_secret = api_secret or os.getenv("API_SECRET")
        # Tokens can be supplied via environment variables to avoid repeated REST calls
        self.ws_token = ws_token or os.getenv("KRAKEN_WS_TOKEN")
        self.api_token = api_token or os.getenv("KRAKEN_API_TOKEN")

        self.exchange = None
        if self.api_key and self.api_secret:
            self.exchange = ccxt.kraken(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                }
            )

        self.token: Optional[str] = self.ws_token
        self.public_ws: Optional[WebSocketApp] = None
        self.private_ws: Optional[WebSocketApp] = None

    def get_token(self) -> str:
        """Retrieve WebSocket authentication token via Kraken REST API."""
        if self.token:
            return self.token

        if not self.exchange:
            raise ValueError("API keys required for private websocket")

        params = {}
        if self.api_token:
            params["otp"] = self.api_token

        resp = self.exchange.privatePostGetWebSocketsToken(params)
        self.token = resp["token"]
        return self.token

    def _start_ws(
        self,
        url: str,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        **kwargs,
    ) -> WebSocketApp:
        """Start a WebSocket connection with basic logging callbacks."""

        def default_on_message(ws, message):
            logger.info("WS message: %s", message)

        def default_on_error(ws, error):
            logger.error("WS error: %s", error)

        def default_on_close(ws, close_status_code, close_msg):
            logger.info("WS closed: %s %s", close_status_code, close_msg)

        on_message = on_message or default_on_message
        on_error = on_error or default_on_error
        on_close = on_close or default_on_close

        ws = WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            **kwargs,
        )
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
