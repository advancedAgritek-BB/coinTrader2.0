import base64
import hashlib
import hmac
import json
import time
import urllib.request
import urllib.parse

DEFAULT_KRAKEN_URL = "https://api.kraken.com"
PATH = "/0/private/GetWebSocketsToken"


def get_ws_token(api_key: str, private_key: str, otp: str | None = None, environment: str = DEFAULT_KRAKEN_URL) -> str:
    """Return a WebSocket authentication token from Kraken."""
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

    req = urllib.request.Request(
        environment + PATH,
        data=encoded_body.encode(),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    token = data.get("result", {}).get("token") or data.get("token")
    if not token:
        raise ValueError("Token not found in response")
    return token
