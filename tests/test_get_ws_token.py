import base64
import hashlib
import hmac
import time
import urllib.parse

import pytest
import requests

from crypto_bot.utils import kraken


class DummyResponse:
    """Simple stand-in for :class:`requests.Response`."""

    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_get_ws_token_success(monkeypatch):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"result": {"token": "abc"}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1)
    secret = base64.b64encode(b"secret").decode()
    token = get_ws_token("key", secret)
    assert token == "abc"
    assert captured["url"].endswith(PATH)

class DummyResp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return {"result": {"token": self.data}}

    def raise_for_status(self):
        pass


def test_get_ws_token(monkeypatch):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"result": {"token": "abc"}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1)
    monkeypatch.setattr(
        kraken, "environment", kraken.DEFAULT_KRAKEN_URL, raising=False
    )

    secret = base64.b64encode(b"secret").decode()
    token = kraken.get_ws_token("key", secret, otp="otp")
    assert token == "abc"

    expected_nonce = "1000"
    expected_body = urllib.parse.urlencode({"nonce": expected_nonce, "otp": "otp"})
    message = kraken.PATH.encode() + hashlib.sha256(
        (expected_nonce + expected_body).encode()
    ).digest()
    expected_sig = base64.b64encode(
        hmac.new(base64.b64decode(secret), message, hashlib.sha512).digest()
    ).decode()

    assert captured["url"].endswith(kraken.PATH)
    assert captured["data"] == expected_body
    assert captured["headers"]["API-Key"] == "key"
    assert captured["headers"]["API-Sign"] == expected_sig
    assert (
        captured["headers"]["Content-Type"]
        == "application/x-www-form-urlencoded"
    )
    assert captured["timeout"] == 15.0


def test_get_ws_token_top_level(monkeypatch):
    monkeypatch.setattr(
        kraken, "environment", kraken.DEFAULT_KRAKEN_URL, raising=False
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: DummyResponse({"token": "abc"}),
    )
    monkeypatch.setattr(time, "time", lambda: 1)
    secret = base64.b64encode(b"secret").decode()
    assert kraken.get_ws_token("key", secret) == "abc"


def test_get_ws_token_missing_token(monkeypatch):
    monkeypatch.setattr(
        kraken, "environment", kraken.DEFAULT_KRAKEN_URL, raising=False
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: DummyResponse({"result": {}}),
    )
    monkeypatch.setattr(time, "time", lambda: 1)
    secret = base64.b64encode(b"secret").decode()
    with pytest.raises(RuntimeError):
        kraken.get_ws_token("key", secret)


def test_get_ws_token_request_failure(monkeypatch):
    monkeypatch.setattr(
        kraken, "environment", kraken.DEFAULT_KRAKEN_URL, raising=False
    )

    def fake_post(*a, **k):
        get_ws_token("key", secret)


def test_get_ws_token_request_failure(monkeypatch):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(time, "time", lambda: 1)
    secret = base64.b64encode(b"secret").decode()
    with pytest.raises(requests.RequestException):
        kraken.get_ws_token("key", secret)
        get_ws_token("key", secret)

    expected_nonce = "1000"
    expected_body = urllib.parse.urlencode({"nonce": expected_nonce})
    message = PATH.encode() + hashlib.sha256((expected_nonce + expected_body).encode()).digest()
    expected_sig = base64.b64encode(
        hmac.new(base64.b64decode(secret), message, hashlib.sha512).digest()
    ).decode()

    assert captured["url"].endswith(PATH)
    assert captured["data"] == expected_body
    assert captured["headers"]["API-Key"] == "key"
    assert captured["headers"]["API-Sign"] == expected_sig
    assert (
        captured["headers"]["Content-Type"]
        == "application/x-www-form-urlencoded"
    )
    assert captured["timeout"] == 15.0
