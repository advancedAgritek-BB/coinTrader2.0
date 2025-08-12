import asyncio
import aiohttp
from crypto_bot.execution import solana_executor


from crypto_bot.utils.telegram import TelegramNotifier


def test_execute_swap_dry_run(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            1,
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
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
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
            data = {"inAmount": 120, "outAmount": 100}
        return DummyResp(data)

    def post(self, url, json=None, timeout=10):
        class PR:
            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        return PR({"swapTransaction": "tx"})


class BadRouteSession(DummySession):
    def get(self, url, params=None, timeout=10):
        if params.get("inputMint") == "SOL":
            data = {"inAmount": 100, "outAmount": 110}
        else:
            data = {"outAmount": 110}  # missing inAmount
        return DummyResp(data)


class DummyJitoSession(DummySession):
    def post(self, url, json=None, headers=None, timeout=10):
        if "jito" in url:
            self.jito_payload = {"url": url, "json": json, "headers": headers}
            class PR:
                def __init__(self, data):
                    self._data = data

                async def json(self):
                    return self._data

                def raise_for_status(self):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    pass

            return PR({"signature": "sig"})
        return super().post(url, json=json, timeout=timeout)


class DummyAsyncClient:
    def __init__(self, url):
        self.url = url
        DummyAsyncClient.instance = self
        self.called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def confirm_transaction(self, sig, commitment=None, sleep_seconds=0.5, last_valid_block_height=None):
        self.called = True
        return {"status": "confirmed"}


def test_execute_swap_skips_on_slippage(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    monkeypatch.setattr(solana_executor, "Client", Client, raising=False)
    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)
    monkeypatch.setattr(solana_executor, "AsyncClient", AC, raising=False)
    async def no_wait(coro, timeout):
        return await coro
    monkeypatch.setattr(asyncio, "wait_for", no_wait)

    notifier = DummyNotifier()
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            TelegramNotifier("t", "c"),
            notifier=notifier,
            dry_run=False,
            config={"max_slippage_pct": 0.05, "confirm_execution": True},
        )
    )
    assert res == {}


def test_slippage_calc_failure(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: BadRouteSession())
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)

    notifier = DummyNotifier()
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=notifier,
            dry_run=False,
            config={"max_slippage_pct": 0.05, "confirm_execution": True},
        )
    )

    assert res == {
        "token_in": "SOL",
        "token_out": "USDC",
        "amount": 100,
        "tx_hash": "h",
        "route": {"inAmount": 100, "outAmount": 110},
        "status": "confirmed",
    }


def test_swap_no_message_when_disabled(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    calls = {"count": 0}

    monkeypatch.setattr(
        "crypto_bot.utils.telegram.send_message_sync",
        lambda *a, **k: calls.__setitem__("count", calls["count"] + 1),
    )
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)

    asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=TelegramNotifier(False, "t", "c"),
            dry_run=False,
            config={"max_slippage_pct": 0.05, "confirm_execution": True},
        )
    )

    assert calls["count"] == 0


class EmptyResp:
    def __init__(self):
        self._data = {"data": []}

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class EmptySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, *a, **k):
        return EmptyResp()

    def post(self, *a, **k):
        return EmptyResp()


def test_execute_swap_no_routes(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: EmptySession())
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            TelegramNotifier("t", "c"),
            notifier=DummyNotifier(),
            dry_run=False,
            config={"confirm_execution": True},
        )
    )
    assert res == {}


class ErrorSession(DummySession):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get(self, url, params=None, timeout=10):
        self.calls += 1
        if self.calls == 1:
            raise solana_executor.aiohttp.ClientError("boom")
        return super().get(url, params=params, timeout=timeout)

    def post(self, url, json=None, timeout=10):
        class R:
            async def json(self):
                return {"swapTransaction": "dHg="}

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

        return R()


def test_execute_swap_quote_retry(monkeypatch):
    import importlib, sys
    sys.modules["aiohttp"] = importlib.import_module("aiohttp")
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    session = ErrorSession()
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: session)
    NetworkError = type("ClientError", (Exception,), {})
    monkeypatch.setattr(solana_executor.aiohttp, "ClientError", NetworkError, raising=False)
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)

    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=DummyNotifier(),
            dry_run=False,
            max_retries=2,
            config={"confirm_execution": True},
        )
    )

    assert res.get("tx_hash") == "h"
    assert delays


class DummyMempool:
    def __init__(self, fee):
        self.fee = fee

    async def fetch_priority_fee(self):
        return self.fee

    async def is_suspicious(self, threshold):
        return False


def test_fee_abort(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: EmptySession())
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

    monitor = DummyMempool(0.01)
    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            TelegramNotifier("t", "c"),
            notifier=DummyNotifier(),
            dry_run=False,
            mempool_monitor=monitor,
            mempool_cfg={"enabled": True},
            config={"take_profit_pct": 0.08, "confirm_execution": True},
        )
    )
    assert res == {}


def test_execute_swap_jito(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.setattr(TelegramNotifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.Notifier, "notify", lambda self, text: None)
    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    session = DummyJitoSession()
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: session)
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

        def serialize(self):
            return b"signed"

    class Client:
        def __init__(self, *a, **k):
            pass

        def send_transaction(self, tx, kp):
            return {"result": "h"}

    import sys, types

    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", DummyAsyncClient, raising=False)

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            TelegramNotifier("t", "c"),
            notifier=DummyNotifier(),
            dry_run=False,
            jito_key="KEY",
            config={"max_slippage_pct": 20, "confirm_execution": True},
        )
    )
    assert res == {
        "token_in": "SOL",
        "token_out": "USDC",
        "amount": 100,
        "tx_hash": "sig",
        "route": {"inAmount": 100, "outAmount": 110},
        "status": "confirmed",
    }
    assert hasattr(session, "jito_payload")
    assert DummyAsyncClient.instance.called


def test_swap_paused_on_suspicious(monkeypatch):
    class DummyMonitor:
        async def fetch_priority_fee(self):
            return 0.0

        async def is_suspicious(self, threshold):
            return True

    monkeypatch.setattr(solana_executor.TelegramNotifier, "notify", lambda *a, **k: None)
    monitor = DummyMonitor()

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            1,
            notifier=DummyNotifier(),
            dry_run=False,
            mempool_monitor=monitor,
            mempool_cfg={"enabled": True, "action": "pause", "suspicious_fee_threshold": 0},
            config={"confirm_execution": True},
        )
    )
    assert res.get("paused") is True


def test_confirm_transaction_called(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
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
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", DummyAsyncClient, raising=False)

    asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=TelegramNotifier(False, "t", "c"),
            dry_run=False,
            config={"max_slippage_pct": 20},
        )
    )

    assert DummyAsyncClient.instance.called


def test_execute_swap_confirms_with_retry(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
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

        def serialize(self):
            return b"signed"

    class Client:
        def __init__(self, *a, **k):
            Client.instance = self
            self.calls = 0

        def send_transaction(self, tx, kp):
            self.calls += 1
            return {"result": "h"}

    class AC:
        calls = 0

        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, *a, **k):
            AC.calls += 1
            if AC.calls == 1:
                raise Exception("fail")
            return {"status": "confirmed"}

    import sys, types

    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)
    monkeypatch.setattr(solana_executor, "AsyncClient", AC, raising=False)

    async def no_sleep(*a, **k):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=DummyNotifier(),
            dry_run=False,
            config={"max_slippage_pct": 20, "confirm_execution": True},
        )
    )

    assert AC.calls == 2
    assert Client.instance.calls == 1


def test_keyring_fallback(monkeypatch):
    monkeypatch.setenv("SOLANA_RPC_URL", "http://dummy")
    monkeypatch.delenv("SOLANA_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(solana_executor.aiohttp, "ClientSession", lambda: DummySession())

    import sys, types
    keyring_stub = types.SimpleNamespace(get_password=lambda s, k: "[1,2,3,4]")
    monkeypatch.setitem(sys.modules, "keyring", keyring_stub)
    monkeypatch.setattr(solana_executor, "keyring", keyring_stub, raising=False)

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

    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))
    sys.modules.setdefault("solana.rpc.async_api", types.ModuleType("solana.rpc.async_api"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    monkeypatch.setattr(sys.modules["solana.transaction"], "Transaction", Tx, raising=False)
    monkeypatch.setattr(sys.modules["solana.rpc.api"], "Client", Client, raising=False)

    class AC:
        def __init__(self, url):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def confirm_transaction(self, sig, commitment=None, sleep_seconds=0.5, last_valid_block_height=None):
            return {}

    monkeypatch.setattr(sys.modules["solana.rpc.async_api"], "AsyncClient", AC, raising=False)

    res = asyncio.run(
        solana_executor.execute_swap(
            "SOL",
            "USDC",
            100,
            notifier=DummyNotifier(),
            dry_run=False,
            config={"confirm_execution": True},
        )
    )

    assert res.get("tx_hash") == "h"

    assert res == {
        "token_in": "SOL",
        "token_out": "USDC",
        "amount": 100,
        "tx_hash": "h",
        "route": {"inAmount": 100, "outAmount": 110},
        "status": "confirmed",
    }
    assert DummyAsyncClient.instance.called
