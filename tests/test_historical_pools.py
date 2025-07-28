from backtest.historical_pools import fetch_pool_history
from crypto_bot.solana.watcher import NewPoolEvent


class DummyResp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_fetch_pool_history(monkeypatch):
    data = {
        "result": {
            "pools": [
                {
                    "address": "P1",
                    "tokenMint": "M1",
                    "creator": "C1",
                    "liquidity": 5.5,
                    "txCount": 2,
                }
            ]
        }
    }

    def dummy_post(url, json=None, timeout=10):
        dummy_post.payload = json
        return DummyResp(data)

    monkeypatch.setattr('backtest.historical_pools.requests.post', dummy_post)

    events = fetch_pool_history(0, 1, "http://test")
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, NewPoolEvent)
    assert event.pool_address == "P1"
    assert event.token_mint == "M1"
    assert event.creator == "C1"
    assert event.liquidity == 5.5
    assert event.tx_count == 2
    assert dummy_post.payload["method"] == "dex.getNewPools"
