import base64
import hashlib
import hmac
import os
import time
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from typing import Any

try:  # optional dependency
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxt = None  # type: ignore

DEFAULT_KRAKEN_URL = "https://api.kraken.com"
PATH = "/0/private/GetWebSocketsToken"

# Flag controlling whether the private WebSocket API should be used. Tests
# toggle this to exercise fallback behaviour in ``cex_executor``.
use_private_ws = True


KrakenClient = Any  # type: ignore
_client: KrakenClient | None = None


def _build_session(pool_maxsize: int = 100, retries: int = 3) -> requests.Session:
    """Return a ``requests.Session`` with an expanded connection pool."""

    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.3, raise_on_status=False)
    adapter = HTTPAdapter(
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
        max_retries=retry,
    )
    session.mount("https://api.kraken.com", adapter)
    session.mount("http://api.kraken.com", adapter)
    session.headers.update({"User-Agent": "coinTrader2.0"})
    return session


def get_http_session():
    from .market_loader import get_http_session as _get_http_session

    return _get_http_session()


def get_client(
    api_key: str | None = None,
    api_secret: str | None = None,
    pool_maxsize: int | None = None,
) -> KrakenClient:
    """Return a singleton ``ccxt.kraken`` client instance."""

    if ccxt is None:  # pragma: no cover - optional dependency
        raise RuntimeError("ccxt Kraken client not available")

    global _client
    if _client is None:
        exchange_cls = getattr(ccxt, "kraken", None)
        if exchange_cls is None:
            raise RuntimeError("Kraken exchange not supported by ccxt")
        _client = exchange_cls(
            {
                "apiKey": api_key or os.getenv("API_KEY"),
                "secret": api_secret or os.getenv("API_SECRET"),
                "enableRateLimit": True,
                "session": _build_session(pool_maxsize or 100),
            }
        )
    return _client


def get_ws_token(
    api_key: str,
    api_secret: str,
    otp: str | None = None,
    environment: str = DEFAULT_KRAKEN_URL,
    timeout: float = 15.0,
) -> str:
    """Return a WebSocket authentication token from Kraken.

    The function performs a signed ``POST`` request to ``environment`` using
    :mod:`requests` and returns the token found either at ``result.token`` or the
    top-level ``token`` key. Network failures raise the original
    :class:`requests.RequestException` while missing tokens and malformed
    responses raise :class:`RuntimeError`.
    """

    if otp is None:
        otp = os.getenv("KRAKEN_OTP")

    nonce = str(int(time.time() * 1000))
    body = {"nonce": nonce}
    if otp:
        body["otp"] = otp
    encoded_body = urllib.parse.urlencode(body)

    message = PATH.encode() + hashlib.sha256((nonce + encoded_body).encode()).digest()
    signature = base64.b64encode(
        hmac.new(base64.b64decode(api_secret), message, hashlib.sha512).digest()
    ).decode()

    headers = {
        "API-Key": api_key,
        "API-Sign": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    session = get_http_session()
    try:
        resp = session.post(
            f"{environment}{PATH}",
            data=encoded_body,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        raise
    except ValueError as exc:  # json decoding error
        raise RuntimeError("Failed to parse WebSocket token response") from exc

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    token = data.get("result", {}).get("token") or data.get("token")
    if not token:
        raise RuntimeError("Token not found in response")
    return token
