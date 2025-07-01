import asyncio
from crypto_bot.execution.solana_executor import execute_swap


def test_execute_swap_dry_run():
    res = asyncio.run(
        execute_swap('SOL', 'USDC', 1, 'token', 'chat', dry_run=True)
    )
    assert res['token_in'] == 'SOL'
    assert res['token_out'] == 'USDC'
    assert res['amount'] == 1
    assert res['tx_hash'] == 'DRYRUN'
