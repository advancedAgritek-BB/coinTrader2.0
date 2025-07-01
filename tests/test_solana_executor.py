import asyncio
from crypto_bot.execution import solana_executor


def test_execute_swap_dry_run(monkeypatch):
    monkeypatch.setattr(solana_executor, "send_message", lambda *a, **k: None)
    res = asyncio.run(
        solana_executor.execute_swap("SOL", "USDC", 1, "token", "chat", dry_run=True)
    )
    assert res == {
        "token_in": "SOL",
        "token_out": "USDC",
        "amount": 1,
        "tx_hash": "DRYRUN",
    }
