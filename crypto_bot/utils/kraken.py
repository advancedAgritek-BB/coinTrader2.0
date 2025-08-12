import base64
import hashlib
import hmac
import os
import time
import urllib.parse

import requests

DEFAULT_KRAKEN_URL = "https://api.kraken.com"
PATH = "/0/private/GetWebSocketsToken"


def get_ws_token(
    api_key: str,
    api_secret: str,
    otp: str | None = None,
    timeout: float = 15.0,
) -> str:
    """Return a WebSocket authentication token from Kraken."""

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

    try:
        resp = requests.post(
            DEFAULT_KRAKEN_URL + PATH,
            data=encoded_body,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("Failed to fetch WebSocket token") from exc

    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")

    token = data.get("result", {}).get("token") or data.get("token")
    if not token:
        raise RuntimeError("Token not found in response")
    return token
