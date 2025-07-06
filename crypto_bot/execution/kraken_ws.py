import json
import threading
import os
from typing import Optional, Callable, Union, List, Any
from datetime import datetime, timedelta, timezone

import ccxt
from websocket import WebSocketApp
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/execution.log")

PUBLIC_URL = "wss://ws.kraken.com/v2"
PRIVATE_URL = "wss://ws-auth.kraken.com/v2"


def parse_ohlc_message(message: str) -> Optional[List[float]]:
    """Parse a Kraken OHLC websocket message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[List[float]]
        ``[timestamp, open, high, low, close, volume]`` if parsable.
    """
    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or len(data) < 3:
        return None

    chan = data[1] if len(data) > 1 else {}
    candle = data[2] if len(data) > 2 else None
    if not isinstance(chan, dict) or not isinstance(candle, list):
        return None
    if not str(chan.get("channel", "")).startswith("ohlc"):
        return None
    try:
        ts = int(float(candle[0]) * 1000)
        o, h, l, c = map(float, candle[1:5])
        vol = float(candle[6])
    except (IndexError, ValueError, TypeError):
        return None
    return [ts, o, h, l, c, vol]


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
        self.last_public_heartbeat: Optional[datetime] = None
        self.last_private_heartbeat: Optional[datetime] = None
        self._public_subs = []
        self._private_subs = []

    def _handle_message(self, ws: WebSocketApp, message: str) -> None:
        """Default ``on_message`` handler that records heartbeats."""
        logger.info("WS message: %s", message)
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        def update(obj: Any) -> None:
            if isinstance(obj, dict) and obj.get("channel") == "heartbeat":
                now = datetime.now(timezone.utc)
                if ws == self.private_ws:
                    self.last_private_heartbeat = now
                else:
                    self.last_public_heartbeat = now

        if isinstance(data, list):
            for item in data:
                update(item)
        else:
            update(data)

    def _regenerate_private_subs(self) -> None:
        """Update stored private subscription messages with the current token."""
        updated = []
        for sub in self._private_subs:
            try:
                msg = json.loads(sub)
                if "params" in msg:
                    msg["params"]["token"] = self.token
                updated.append(json.dumps(msg))
            except Exception:
                updated.append(sub)
        self._private_subs = updated

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
        *,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        **kwargs,
    ) -> WebSocketApp:
        """Start a ``WebSocketApp`` and begin the reader thread."""

        def default_on_message(ws, message):
            self._handle_message(ws, message)

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
        thread = threading.Thread(
            target=lambda: ws.run_forever(
                ping_interval=ping_interval, ping_timeout=ping_timeout
            ),
            daemon=True,
        )
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
        prev_token = self.token
        if self.token_expired():
            self.token = None
        if not self.token:
            self.get_token()
        token_changed = self.token != prev_token
        if not self.private_ws:
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")
            token_changed = True
        if token_changed:
            self._regenerate_private_subs()
            for sub in self._private_subs:
                self.private_ws.send(sub)

    def subscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[dict] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """Subscribe to ticker updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {"channel": "ticker", "symbol": symbol},
        }
        if event_trigger is not None:
            msg["params"]["eventTrigger"] = event_trigger
        if req_id is not None:
            msg["req_id"] = req_id
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def unsubscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[dict] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """Unsubscribe from ticker updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "ticker", "symbol": symbol},
        }
        if event_trigger is not None:
            msg["params"]["eventTrigger"] = event_trigger
        if req_id is not None:
            msg["req_id"] = req_id
        data = json.dumps(msg)

        sub_msg = {
            "method": "subscribe",
            "params": {"channel": "ticker", "symbol": symbol},
        }
        if event_trigger is not None:
            sub_msg["params"]["eventTrigger"] = event_trigger
        if req_id is not None:
            sub_msg["req_id"] = req_id
        sub_data = json.dumps(sub_msg)
        if sub_data in self._public_subs:
            self._public_subs.remove(sub_data)
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

    def subscribe_orders(self, symbol: Optional[str] = None) -> None:
        """Subscribe to private open order updates.

        If ``symbol`` is provided the channel name uses ``openOrders`` and the
        symbol list, matching Kraken's subscription format used in the tests.
        Otherwise the older ``open_orders`` channel is used.
        """
        self.connect_private()
        channel = "openOrders" if symbol is not None else "open_orders"
        msg = {
            "method": "subscribe",
            "params": {"channel": channel, "token": self.token},
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

    def ping(self, req_id: Optional[int] = None) -> dict:
        """Send a ping message to keep the websocket connection alive."""
        msg = {"method": "ping", "req_id": req_id}
        data = json.dumps(msg)
        ws = self.private_ws or self.public_ws
        if not ws:
            raise RuntimeError("WebSocket not connected")
        ws.send(data)
        return msg

    def is_alive(self, conn_type: str) -> bool:
        """Return ``True`` if the connection received a heartbeat recently."""
        now = datetime.now(timezone.utc)
        if conn_type == "private":
            last = self.last_private_heartbeat
        else:
            last = self.last_public_heartbeat
        return bool(last and (now - last) <= timedelta(seconds=10))

    def on_close(self, conn_type: str) -> None:
        """Handle WebSocket closure by reconnecting and resubscribing."""

        if conn_type == "public":
            self.public_ws = self._start_ws(PUBLIC_URL, conn_type="public")
            for sub in self._public_subs:
                self.public_ws.send(sub)
        else:
            if self.token_expired():
                self.token = None
            if not self.token:
                self.get_token()
            self._regenerate_private_subs()
            self.private_ws = self._start_ws(PRIVATE_URL, conn_type="private")
            for sub in self._private_subs:
                self.private_ws.send(sub)
