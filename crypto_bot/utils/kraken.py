import base64
import base64
import hashlib
import hmac
import time
import urllib.parse

import requests

DEFAULT_KRAKEN_URL = "https://api.kraken.com"
PATH = "/0/private/GetWebSocketsToken"

# Flag controlling whether the private WebSocket API should be used. Tests
# toggle this to exercise fallback behaviour in ``cex_executor``.
use_private_ws = True


def get_ws_token(
    api_key: str,
    private_key: str,
    otp: str | None = None,
    environment: str = DEFAULT_KRAKEN_URL,
) -> str:
    """Return a WebSocket authentication token from Kraken.

    The function performs a signed ``POST`` request using ``requests`` and
    returns the token found either at ``result.token`` or the top-level
    ``token`` key. If no token can be extracted a :class:`ValueError` is raised.
    ``requests`` exceptions are allowed to propagate so callers can decide how
    to handle network failures.
    """

    nonce = str(int(time.time() * 1000))
    body = [("nonce", nonce)]
    if otp:
        body.append(("otp", otp))
    encoded_body = urllib.parse.urlencode(body)

    message = PATH.encode() + hashlib.sha256((nonce + encoded_body).encode()).digest()
    signature = base64.b64encode(
        hmac.new(base64.b64decode(private_key), message, hashlib.sha512).digest()
    ).decode()

    headers = {
        "API-Key": api_key,
        "API-Sign": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = requests.post(environment + PATH, data=encoded_body, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    token = data.get("result", {}).get("token") or data.get("token")
    if not token:
        raise ValueError("Token not found in response")
    return token
