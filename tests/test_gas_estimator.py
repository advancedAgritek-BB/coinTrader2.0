from crypto_bot.utils.gas_estimator import estimate_gas_fee_usd, gas_fee_too_high


def test_estimate_eth_gas(monkeypatch):
    monkeypatch.setenv("MOCK_ETH_GAS_PRICE_WEI", "20000000000")  # 20 gwei
    fee = estimate_gas_fee_usd("ethereum", 21000, 2000)
    assert fee == 20000000000 * 21000 / 1_000_000_000_000_000_000 * 2000


def test_estimate_solana_gas(monkeypatch):
    monkeypatch.setenv("MOCK_SOLANA_FEE_LAMPORTS", "5000")
    fee = estimate_gas_fee_usd("solana", 1, 20)
    assert fee == 5000 / 1_000_000_000 * 20


def test_gas_fee_limit(monkeypatch):
    monkeypatch.setenv("MOCK_ETH_GAS_PRICE_WEI", "20000000000")
    # fee is ~0.84 USD for 21000 gas at $2000 ETH
    assert gas_fee_too_high("ethereum", 1000, 21000, 0.5, 2000) is False
    assert gas_fee_too_high("ethereum", 10, 21000, 0.5, 2000) is True
