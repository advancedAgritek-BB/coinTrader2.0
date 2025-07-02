import json
import threading
import os
from typing import Optional, Callable, Union, List
from datetime import datetime, timedelta, timezone

import ccxt
from websocket import WebSocketApp
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/execution.log")

PUBLIC_URL = "wss://ws.kraken.com/v2"
PRIVATE_URL = "wss://ws-auth.kraken.com/v2"


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
        self.token_created: Optional[datetime] = None
        if self.token:
            self.token_created = datetime.now(timezone.utc)
        self.public_ws: Optional[WebSocketApp] = None
        self.private_ws: Optional[WebSocketApp] = None
        self._public_subs = []
        self._private_subs = []

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
        self.token_created = datetime.now(timezone.utc)
        return self.token

    def _start_ws(
        self,
        url: str,
        conn_type: Optional[str] = None,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        **kwargs,
    ) -> WebSocketApp:
        """Start a ``WebSocketApp`` and begin the reader thread."""

        def default_on_message(ws, message):
            logger.info("WS message: %s", message)

        def default_on_error(ws, error):
            logger.error("WS error: %s", error)

        def default_on_close(ws, close_status_code, close_msg):
            logger.info("WS closed: %s %s", close_status_code, close_msg)

        on_message = on_message or default_on_message
        on_error = on_error or default_on_error

        def _on_close(ws, close_status_code, close_msg):
            if on_close:
                on_close(ws, close_status_code, close_msg)
            else:
                default_on_close(ws, close_status_code, close_msg)
            if conn_type:
                self.on_close(conn_type)

        ws = WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=_on_close,
            **kwargs,
        )
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        return ws

    def token_expired(self) -> bool:
        """Return True if the authentication token is older than 14 minutes."""
        if not self.token_created:
            return False
        return datetime.now(timezone.utc) - self.token_created > timedelta(minutes=14)


    def connect_public(self) -> None:
        if not self.public_ws:
            self.public_ws = self._start_ws(PUBLIC_URL, conn_type="public")

    def connect_private(self) -> None:
        if self.token_expired():
            self.token = None
        if not self.token:
            self.get_token()
        if not self.private_ws:
            if not self.token:
                self.get_token()
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")

    def subscribe_ticker(self, symbol: Union[str, List[str]]) -> None:
        """Subscribe to ticker updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {"channel": "ticker", "symbol": symbol},
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_trades(self, symbol: Union[str, List[str]]) -> None:
        """Subscribe to trade updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": symbol},
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_orders(self) -> None:
        """Subscribe to private open order updates."""
        """Subscribe to the authenticated ``openOrders`` channel."""
        self.connect_private()
        msg = {
            "method": "subscribe",
            "params": {"channel": "open_orders", "token": self.token},
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def add_order(
        self,
        symbol: Union[str, List[str]],
        side: str,
        volume: float,
        ordertype: str = "market",
    ) -> dict:
        """Send an add_order request via the private websocket."""
        self.connect_private()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "add_order",
            "params": {
                "symbol": symbol,
                "type": side,
                "ordertype": ordertype,
                "volume": str(volume),
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def cancel_order(self, txid: str) -> dict:
        self.connect_private()
        msg = {
            "method": "cancel_order",
            "params": {"txid": txid, "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def cancel_all_orders(self) -> dict:
        self.connect_private()
        msg = {
            "method": "cancel_all_orders",
            "params": {"token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def open_orders(self) -> dict:
        self.connect_private()
        msg = {
            "method": "open_orders",
            "params": {"token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        return msg

    def on_close(self, conn_type: str) -> None:
        """Handle WebSocket closure by reconnecting and resubscribing."""

        if conn_type == "public":
            self.public_ws = self._start_ws(PUBLIC_URL, conn_type="public")
            for sub in self._public_subs:
                self.public_ws.send(sub)
        else:
            if not self.token:
                self.get_token()
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")
            for sub in self._private_subs:
                self.private_ws.send(sub)
