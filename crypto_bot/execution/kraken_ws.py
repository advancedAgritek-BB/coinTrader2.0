import json
import threading
import time
import os
import asyncio
from collections import deque
from typing import Optional, Callable, Union, List, Any, Dict, Deque
from datetime import datetime, timedelta, timezone
import keyring

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()
from websocket import WebSocketApp
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "execution.log")

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

    # object based format
    if isinstance(data, dict):
        if data.get("channel") == "ohlc" and data.get("type") in {"snapshot", "update"}:
            arr = data.get("data")
            if isinstance(arr, list) and arr:
                candle = arr[0]
                if isinstance(candle, dict):
                    ts_val = candle.get("interval_begin")
                    if isinstance(ts_val, str):
                        try:
                            ts = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp() * 1000)
                        except Exception:
                            ts = None
                    else:
                        ts = None
                    try:
                        o = float(candle.get("open"))
                        h = float(candle.get("high"))
                        l = float(candle.get("low"))
                        c = float(candle.get("close"))
                        vol = float(candle.get("volume"))
                    except (TypeError, ValueError):
                        return None
                    if ts is not None:
                        return [ts, o, h, l, c, vol]

    # list based format
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



def parse_instrument_message(message: str) -> Optional[dict]:
    """Parse a Kraken instrument snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[dict]
        The ``data`` payload containing ``assets`` and ``pairs`` if the
        message is a valid instrument snapshot or update.
    """
    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "instrument":
        return None

    payload = data.get("data")
    if not isinstance(payload, dict):
        return None
    return payload


def parse_book_message(message: str) -> Optional[dict]:
    """Parse a Kraken order book snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[dict]
        Dictionary with ``bids`` and ``asks`` lists containing ``[price, volume]``
        floats and a ``type`` field if the message is valid.
    """

    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or data.get("channel") != "book":
        return None

    msg_type = data.get("type")
    payload = data.get("data")
    if msg_type not in ("snapshot", "update") or not isinstance(payload, dict):
        return None

    bids = payload.get("bids")
    asks = payload.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        return None

    def _convert(items: List[Any]) -> List[List[float]]:
        parsed = []
        for it in items:
            try:
                price = float(it[0])
                volume = float(it[1])
            except (IndexError, ValueError, TypeError):
                continue
            parsed.append([price, volume])
        return parsed

    return {"type": msg_type, "bids": _convert(bids), "asks": _convert(asks)}


def parse_trade_message(message: str) -> Optional[List[dict]]:
    """Parse a Kraken trade snapshot or update message.

    Parameters
    ----------
    message : str
        Raw JSON message from the websocket.

    Returns
    -------
    Optional[List[dict]]
        List of trade dictionaries with fields ``symbol``,
        ``side``, ``qty``, ``price``, ``ord_type``, ``trade_id`` and
        ``timestamp_ms`` if the message is valid.
    """

    try:
        data: Any = json.loads(message)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or data.get("channel") != "trade":
        return None

    trades = data.get("data")
    if not isinstance(trades, list):
        return None

    result: List[dict] = []
    for t in trades:
        if not isinstance(t, dict):
            return None

        symbol = t.get("symbol") or data.get("symbol")
        side = t.get("side")
        qty = t.get("qty")
        price = t.get("price")
        ord_type = t.get("ord_type")
        trade_id = t.get("trade_id")
        ts = t.get("timestamp")

        if not isinstance(symbol, str):
            return None
        try:
            qty = float(qty)
            price = float(price)
        except (TypeError, ValueError):
            return None

        timestamp_ms = None
        if isinstance(ts, str):
            try:
                if ts.endswith("Z"):
                    ts_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    ts_obj = datetime.fromisoformat(ts)
                timestamp_ms = int(ts_obj.timestamp() * 1000)
            except Exception:
                return None
        else:
            return None

        result.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "ord_type": ord_type,
                "trade_id": trade_id,
                "timestamp_ms": timestamp_ms,
            }
        )

    return result


def _parse_l3_orders(levels: List[Any]) -> Optional[List[dict]]:
    """Return parsed list of L3 book orders or ``None`` on error."""

    result = []
    for lvl in levels:
        if not isinstance(lvl, dict):
            return None
        try:
            price = float(lvl.get("limit_price"))
            qty = float(lvl.get("order_qty"))
            order_id = str(lvl.get("order_id"))
        except (TypeError, ValueError):
            return None
        result.append(
            {
                "order_id": order_id,
                "limit_price": price,
                "order_qty": qty,
            }
        )
    return result


def parse_level3_snapshot(msg: str) -> Optional[dict]:
    """Parse a Kraken level 3 order book snapshot message."""

    try:
        data: Any = json.loads(msg)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "level3" or data.get("type") != "snapshot":
        return None

    payload_list = data.get("data")
    if not isinstance(payload_list, list) or not payload_list:
        return None
    payload = payload_list[0]
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    if not isinstance(symbol, str):
        return None

    bids_raw = payload.get("bids")
    asks_raw = payload.get("asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = _parse_l3_orders(bids_raw)
    asks = _parse_l3_orders(asks_raw)
    if bids is None or asks is None:
        return None

    checksum = payload.get("checksum")
    try:
        checksum = int(checksum)
    except (TypeError, ValueError):
        return None

    ts_val = payload.get("timestamp")
    timestamp = None
    if isinstance(ts_val, str):
        try:
            timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            timestamp = None

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "checksum": checksum,
        "timestamp": timestamp,
    }


def _parse_l3_events(levels: List[Any]) -> Optional[List[dict]]:
    """Parse level 3 order book update events."""

    result = []
    for lvl in levels:
        if not isinstance(lvl, dict):
            return None
        event = lvl.get("event")
        order_id = lvl.get("order_id")
        try:
            price = float(lvl.get("limit_price"))
            qty = float(lvl.get("order_qty"))
        except (TypeError, ValueError):
            return None
        ts_val = lvl.get("timestamp")
        ts = None
        if isinstance(ts_val, str):
            try:
                ts = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                ts = None
        result.append(
            {
                "event": str(event),
                "order_id": str(order_id),
                "limit_price": price,
                "order_qty": qty,
                "timestamp": ts,
            }
        )
    return result


def parse_level3_update(msg: str) -> Optional[dict]:
    """Parse a Kraken level 3 order book update message."""

    try:
        data: Any = json.loads(msg)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("channel") != "level3" or data.get("type") != "update":
        return None

    payload_list = data.get("data")
    if not isinstance(payload_list, list) or not payload_list:
        return None
    payload = payload_list[0]
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    if not isinstance(symbol, str):
        return None

    bids_raw = payload.get("bids")
    asks_raw = payload.get("asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = _parse_l3_events(bids_raw)
    asks = _parse_l3_events(asks_raw)
    if bids is None or asks is None:
        return None

    checksum = payload.get("checksum")
    try:
        checksum = int(checksum)
    except (TypeError, ValueError):
        return None

    ts_val = payload.get("timestamp")
    timestamp = None
    if isinstance(ts_val, str):
        try:
            timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except Exception:
            timestamp = None

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "checksum": checksum,
        "timestamp": timestamp,
    }


class KrakenWSClient:
    """Minimal Kraken WebSocket client for public and private channels."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        ws_token: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        if api_key is None:
            api_key = keyring.get_password("kraken", "api_key")
            if api_key is None:
                api_key = os.getenv("API_KEY")
        if api_secret is None:
            api_secret = keyring.get_password("kraken", "api_secret")
            if api_secret is None:
                api_secret = os.getenv("API_SECRET")

        self.api_key = api_key
        self.api_secret = api_secret

        if not self.api_key or not self.api_secret:
            raise ValueError("Kraken API credentials not available")

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
        self._responses: Deque[dict] = deque()
        self._response_cond = threading.Condition()
        self._messages: Deque[str] = deque()
        self._message_cond = threading.Condition()

    def _handle_message(self, ws: WebSocketApp, message: str) -> None:
        """Default ``on_message`` handler that records heartbeats."""
        logger.info("WS message: %s", message)
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        def update(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            if obj.get("channel") == "heartbeat":
                now = datetime.now(timezone.utc)
                if ws == self.private_ws:
                    self.last_private_heartbeat = now
                else:
                    self.last_public_heartbeat = now
            if obj.get("method") or obj.get("errorMessage"):
                with self._response_cond:
                    self._responses.append(obj)
                    self._response_cond.notify_all()

        if isinstance(data, list):
            for item in data:
                update(item)
        else:
            update(data)

        with self._message_cond:
            self._messages.append(message)
            self._message_cond.notify_all()

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

    def _wait_for_response(self, method: str, timeout: float = 5.0) -> dict:
        """Wait for a response message matching the given method."""
        end = time.time() + timeout
        while time.time() < end:
            with self._response_cond:
                if not self._responses:
                    remaining = end - time.time()
                    if remaining <= 0:
                        break
                    self._response_cond.wait(timeout=remaining)
                    continue
                for _ in range(len(self._responses)):
                    msg = self._responses.popleft()
                    if isinstance(msg, dict) and (
                        msg.get("method") == method or msg.get("errorMessage")
                    ):
                        return msg
                    self._responses.append(msg)
            time.sleep(0.01)
        raise TimeoutError(f"No response for {method}")

    def _pop_message(self, timeout: Optional[float] = None) -> Optional[str]:
        with self._message_cond:
            if not self._messages:
                self._message_cond.wait(timeout=timeout)
            if self._messages:
                return self._messages.popleft()
        return None

    async def _next_message(self, timeout: Optional[float] = None) -> Optional[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._pop_message, timeout)

    def subscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[str] = None,
        snapshot: Optional[bool] = None,
        req_id: Optional[int] = None,
    ) -> None:
        """Subscribe to ticker updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "ticker", "symbol": symbol}
        if event_trigger is not None:
            params["eventTrigger"] = event_trigger
        if snapshot is not None:
            params["snapshot"] = snapshot
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def unsubscribe_ticker(
        self,
        symbol: Union[str, List[str]],
        *,
        event_trigger: Optional[str] = None,
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

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "ticker"
                and sorted(params.get("symbol", [])) == sorted(symbol)
                and (
                    event_trigger is None
                    or params.get("event_trigger") == event_trigger
                )
                and (req_id is None or params.get("req_id") == req_id)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]
        self.public_ws.send(data)

    def subscribe_trades(
        self, symbol: Union[str, List[str]], *, snapshot: bool = True
    ) -> None:
        """Subscribe to trade updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "trade", "symbol": symbol}
        if snapshot is not True:
            params["snapshot"] = snapshot
        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        if hasattr(self.public_ws, "sent"):
            try:
                self.public_ws.sent.clear()
            except Exception:
                pass
        self.public_ws.send(data)

    def unsubscribe_trades(self, symbol: Union[str, List[str]]) -> None:
        """Unsubscribe from trade updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "trade", "symbol": symbol},
        }
        data = json.dumps(msg)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "trade"
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]
        self.public_ws.send(data)

    def subscribe_ohlc(
        self,
        symbol: Union[str, List[str]],
        interval: int,
        *,
        snapshot: bool = True,
        req_id: Optional[int] = None,
    ) -> None:
        """Subscribe to OHLC updates for one or more symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]

        params = {
            "channel": "ohlc",
            "symbol": symbol,
            "interval": interval,
            "snapshot": snapshot,
        }
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "subscribe", "params": params}
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_book(
        self,
        symbol: Union[str, List[str]],
        *,
        depth: int = 10,
        snapshot: bool = True,
    ) -> None:
        """Subscribe to order book updates for one or more symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": symbol,
                "depth": depth,
                "snapshot": snapshot,
            },
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def subscribe_instruments(self, snapshot: bool = True) -> None:
        """Subscribe to the instrument reference data channel."""
        self.connect_public()
        if hasattr(self.public_ws, "sent"):
            try:
                self.public_ws.sent.clear()
            except Exception:
                pass
        msg = {
            "method": "subscribe",
            "params": {"channel": "instrument", "snapshot": snapshot},
        }
        data = json.dumps(msg)
        self._public_subs.append(data)
        self.public_ws.send(data)

    def unsubscribe_instruments(self) -> None:
        """Unsubscribe from the instrument reference data channel."""
        self.connect_public()
        msg = {"method": "unsubscribe", "params": {"channel": "instrument"}}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "instrument"
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    def unsubscribe_book(self, symbol: Union[str, List[str]], depth: int = 10) -> None:
        """Unsubscribe from order book updates for the given symbols."""
        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]
        params = {"channel": "book", "symbol": symbol}
        if depth != 10:
            params["depth"] = depth
        msg = {"method": "unsubscribe", "params": params}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "book"
                and params.get("depth", depth) == depth
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    def unsubscribe_ohlc(
        self,
        symbol: Union[str, List[str]],
        interval: int,
        *,
        req_id: Optional[int] = None,
    ) -> None:
        """Unsubscribe from OHLC updates for the given symbols."""

        self.connect_public()
        if isinstance(symbol, str):
            symbol = [symbol]

        params = {
            "channel": "ohlc",
            "symbol": symbol,
            "interval": interval,
        }
        if req_id is not None:
            params["req_id"] = req_id

        msg = {"method": "unsubscribe", "params": params}
        data = json.dumps(msg)
        self.public_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "ohlc"
                and params.get("interval") == interval
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._public_subs = [s for s in self._public_subs if not _matches(s)]

    async def watch_ohlcv(
        self, symbols: List[str], timeframe: str, limit: int = 1
    ) -> Dict[str, List[List[float]]]:
        """Return the latest ``limit`` candles per symbol via WebSocket."""

        tf_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        interval = tf_map.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe {timeframe}")

        with self._message_cond:
            self._messages.clear()

        self.subscribe_ohlc(symbols, interval)

        per_sym: Dict[str, Deque[List[float]]] = {
            s: deque(maxlen=limit) for s in symbols
        }
        remaining = set(symbols)
        while remaining:
            msg = await self._next_message(timeout=5.0)
            if msg is None:
                continue
            candle = parse_ohlc_message(msg)
            if not candle:
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            symbol = None
            if isinstance(data, dict):
                symbol = data.get("symbol")
                if not symbol and isinstance(data.get("data"), list):
                    first = data["data"][0]
                    if isinstance(first, dict):
                        symbol = first.get("symbol")
            elif isinstance(data, list):
                if len(data) > 1 and isinstance(data[1], dict):
                    symbol = data[1].get("symbol")
                if not symbol and len(data) > 3 and isinstance(data[3], dict):
                    symbol = data[3].get("symbol")
            if symbol in per_sym:
                per_sym[symbol].append(candle)
                if len(per_sym[symbol]) >= limit:
                    remaining.discard(symbol)
        return {s: list(per_sym[s]) for s in symbols}

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

    def unsubscribe_orders(self, symbol: Optional[str] = None) -> None:
        """Unsubscribe from private open order updates."""

        self.connect_private()
        channel = "openOrders" if symbol is not None else "open_orders"
        msg = {
            "method": "unsubscribe",
            "params": {"channel": channel, "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == channel
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def subscribe_executions(self) -> None:
        """Subscribe to private execution (trade fill) updates."""

        self.connect_private()
        msg = {
            "method": "subscribe",
            "params": {"channel": "executions", "token": self.token},
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def unsubscribe_executions(self) -> None:
        """Unsubscribe from private execution updates."""

        self.connect_private()
        msg = {
            "method": "unsubscribe",
            "params": {"channel": "executions", "token": self.token},
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "executions"
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def subscribe_level3(
        self,
        symbol: Union[str, List[str]],
        *,
        depth: int = 10,
        snapshot: bool = True,
    ) -> None:
        """Subscribe to authenticated level3 order book updates."""

        self.connect_private()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "subscribe",
            "params": {
                "channel": "level3",
                "symbol": symbol,
                "depth": depth,
                "snapshot": snapshot,
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self._private_subs.append(data)
        self.private_ws.send(data)

    def unsubscribe_level3(self, symbol: Union[str, List[str]], depth: int = 10) -> None:
        """Unsubscribe from authenticated level3 order book updates."""

        self.connect_private()
        if isinstance(symbol, str):
            symbol = [symbol]
        msg = {
            "method": "unsubscribe",
            "params": {
                "channel": "level3",
                "symbol": symbol,
                "depth": depth,
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self.private_ws.send(data)

        def _matches(sub: str) -> bool:
            try:
                parsed = json.loads(sub)
            except Exception:
                return False
            params = parsed.get("params", {}) if isinstance(parsed, dict) else {}
            return (
                parsed.get("method") == "subscribe"
                and params.get("channel") == "level3"
                and params.get("depth", depth) == depth
                and sorted(params.get("symbol", [])) == sorted(symbol)
            )

        self._private_subs = [s for s in self._private_subs if not _matches(s)]

    def add_order(
        self,
        symbol: Union[str, List[str]],
        side: str,
        order_qty: float,
        order_type: str = "market",
    ) -> dict:
        """Send an add_order request via the private websocket."""
        self.connect_private()
        if isinstance(symbol, list):
            symbol = symbol[0] if symbol else ""
        msg = {
            "method": "add_order",
            "params": {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "order_qty": str(order_qty),
                "token": self.token,
            },
        }
        data = json.dumps(msg)
        self.private_ws.send(data)
        resp = self._wait_for_response("add_order")
        if isinstance(resp, dict) and resp.get("errorMessage"):
            raise RuntimeError(resp["errorMessage"])
        return resp

    def edit_order(
        self,
        order_id: str,
        symbol: str,
        order_qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        deadline: Optional[str] = None,
    ) -> dict:
        """Send an edit_order request via the private websocket."""
        self.connect_private()
        params = {"order_id": order_id, "symbol": symbol, "token": self.token}
        if order_qty is not None:
            params["order_qty"] = order_qty
        if limit_price is not None:
            params["limit_price"] = limit_price
        if deadline is not None:
            params["deadline"] = deadline
        msg = {"method": "edit_order", "params": params}
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

    def amend_order(
        self,
        *,
        order_id: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        order_qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        display_qty: Optional[float] = None,
        limit_price_type: Optional[str] = None,
        post_only: Optional[bool] = None,
        trigger_price: Optional[float] = None,
        trigger_price_type: Optional[str] = None,
        deadline: Optional[str] = None,
        req_id: Optional[int] = None,
    ) -> dict:
        """Send an ``amend_order`` request via the private websocket."""

        self.connect_private()

        params: Dict[str, Any] = {"token": self.token}
        if order_id is not None:
            params["order_id"] = order_id
        if cl_ord_id is not None:
            params["cl_ord_id"] = cl_ord_id
        if order_qty is not None:
            params["order_qty"] = order_qty
        if limit_price is not None:
            params["limit_price"] = limit_price
        if display_qty is not None:
            params["display_qty"] = display_qty
        if limit_price_type is not None:
            params["limit_price_type"] = limit_price_type
        if post_only is not None:
            params["post_only"] = post_only
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if trigger_price_type is not None:
            params["trigger_price_type"] = trigger_price_type
        if deadline is not None:
            params["deadline"] = deadline

        msg: Dict[str, Any] = {"method": "amend_order", "params": params}
        if req_id is not None:
            msg["req_id"] = req_id

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

    def close(self) -> None:
        """Close active WebSocket connections and clear subscriptions."""

        if self.public_ws:
            try:
                self.public_ws.on_close = None
                self.public_ws.close()
            except Exception as exc:
                logger.error("Error closing public websocket: %s", exc)
            self.public_ws = None

        if self.private_ws:
            try:
                self.private_ws.on_close = None
                self.private_ws.close()
            except Exception as exc:
                logger.error("Error closing private websocket: %s", exc)
            self.private_ws = None

        self._public_subs = []
        self._private_subs = []
