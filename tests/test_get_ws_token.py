import base64
import hashlib
import hmac
import json
import urllib.request
import urllib.parse
import time

from crypto_bot.utils.kraken import get_ws_token, PATH

class DummyResp:
    def __init__(self, data):
        self.data = data
    def read(self):
        return json.dumps({"result": {"token": self.data}}).encode()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass

def test_get_ws_token(monkeypatch):
    captured = {}
    def fake_urlopen(req):
        captured['headers'] = dict(req.header_items())
        captured['body'] = req.data.decode()
        return DummyResp('abc')
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(time, 'time', lambda: 1)
    secret = base64.b64encode(b'secret').decode()
    token = get_ws_token('key', secret, otp='otp')
    assert token == 'abc'
    expected_nonce = '1000'
    expected_body = urllib.parse.urlencode({'nonce': expected_nonce, 'otp': 'otp'})
    message = PATH.encode() + hashlib.sha256((expected_nonce + expected_body).encode()).digest()
    expected_sig = base64.b64encode(hmac.new(base64.b64decode(secret), message, hashlib.sha512).digest()).decode()
    # urllib normalizes header keys to title case
    assert captured['headers']['Api-key'] == 'key'
    assert captured['headers']['Api-sign'] == expected_sig
    assert captured['headers']['Content-type'] == 'application/x-www-form-urlencoded'
    assert captured['body'] == expected_body
