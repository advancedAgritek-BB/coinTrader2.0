import asyncio
from crypto_bot.execution import solana_executor


from crypto_bot.utils.telegram import TelegramNotifier


def test_execute_swap_dry_run(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            1,
            notifier=TelegramNotifier("t", "c"),
            dry_run=True,
        )
    )
            TelegramNotifier("t", "c"),
            dry_run=True,
        )
    )

class DummyNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, text: str):
        self.messages.append(text)
        return None


def test_execute_swap_dry_run(monkeypatch):
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    notifier = DummyNotifier()
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL", "USDC", 1, notifier=notifier, dry_run=True
        )
    )
    assert notifier.messages
    assert res == {
        "token_in": "SOL",
        "token_out": "USDC",
        "amount": 1,
        "tx_hash": "DRYRUN",
    }


class DummyResp:
    def __init__(self, data):
        self._data = {"data": [data]}

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, params=None, timeout=10):
        if params.get("inputMint") == "SOL":
            data = {"inAmount": 100, "outAmount": 110}
        else:
            data = {"inAmount": 110, "outAmount": 110}
        return DummyResp(data)

    def post(self, url, json=None, timeout=10):
        return DummyResp({"swapTransaction": "tx"})


def test_execute_swap_skips_on_slippage(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: DummySession())
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "[1,2,3,4]")

    class KP:
        public_key = "k"

        @staticmethod
        def from_secret_key(b):
            return KP()

        def sign(self, tx):
            pass

    class Tx:
        @staticmethod
        def deserialize(raw):
            return Tx()

        def sign(self, kp):
            pass

    class Client:
        def __init__(self, *a, **k):
            pass

        def send_transaction(self, tx, kp):
            return {"result": "h"}

    import sys, types
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    notifier = DummyNotifier()
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=notifier,
            dry_run=False,
            config={"max_slippage_pct": 0.05},
        )
    )
    assert res == {}


def test_swap_no_message_when_disabled(monkeypatch):
    calls = {"count": 0}

    import crypto_bot.utils.telegram_notifier as notifier

    monkeypatch.setattr(notifier, "send_message", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1))
    notifier.TelegramNotifier.configure(False)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: DummySession())
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "[1,2,3,4]")

    class KP:
        public_key = "k"

        @staticmethod
        def from_secret_key(b):
            return KP()

        def sign(self, tx):
            pass

    class Tx:
        @staticmethod
        def deserialize(raw):
            return Tx()

        def sign(self, kp):
            pass

    class Client:
        def __init__(self, *a, **k):
            pass

        def send_transaction(self, tx, kp):
            return {"result": "h"}

    import sys, types
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    try:
        asyncio.run(
            solana_executor.execute_swap(
                "SOL",
                "USDC",
                100,
                "t",
                "c",
                dry_run=False,
                config={"max_slippage_pct": 0.05},
            )
        )
    finally:
        notifier.TelegramNotifier.configure(True)

    assert calls["count"] == 0
