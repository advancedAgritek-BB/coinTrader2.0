import time

from crypto_bot.risk.position_sizing import (
    RiskLimits,
    RiskManager,
    compute_realized_vol,
    record_price,
    size_for_sigma,
)


def test_compute_realized_vol_and_size_for_sigma():
    symbol = "BTCUSDT"
    t = time.time()
    prices = [100.0, 101.0, 99.0, 100.0]
    for i, p in enumerate(prices):
        record_price(symbol, p, t + i)
    vol = compute_realized_vol(symbol, 10)
    assert vol > 0
    size_low_vol = size_for_sigma(10.0, vol, 100.0)
    record_price(symbol, 110.0, t + len(prices))
    vol_high = compute_realized_vol(symbol, 10)
    assert vol_high >= vol
    size_high_vol = size_for_sigma(10.0, vol_high, 100.0)
    assert size_low_vol > size_high_vol


def test_risk_manager_halt_and_cooldown(caplog):
    limits = RiskLimits(starting_equity=1000, daily_loss_limit_pct=0.1, max_consecutive_losses=2, symbol_cooldown_min=0.001)
    rm = RiskManager(limits)
    assert rm.allow_entry("BTC")
    rm.record_result("BTC", -60)
    rm.record_result("ETH", -50)
    assert rm.halt
    assert not rm.allow_entry("BTC")
    assert any("risk_halt" in rec.message for rec in caplog.records)

    limits2 = RiskLimits(starting_equity=1000, daily_loss_limit_pct=0.5, max_consecutive_losses=2, symbol_cooldown_min=0.001)
    rm2 = RiskManager(limits2)
    assert rm2.allow_entry("BTC")
    rm2.record_result("BTC", -1)
    assert rm2.allow_entry("BTC")
    rm2.record_result("BTC", -1)
    assert not rm2.allow_entry("BTC")
    time.sleep(0.1)
    assert rm2.allow_entry("BTC")
