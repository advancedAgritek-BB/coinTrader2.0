from crypto_bot.solana.safety import is_safe
from crypto_bot.solana.watcher import NewPoolEvent


def make_event(**kwargs):
    base = dict(
        pool_address="P",
        token_mint="M",
        creator="C",
        liquidity=100.0,
        tx_count=0,
        freeze_authority="",
        mint_authority="",
        timestamp=0.0,
    )
    base.update(kwargs)
    return NewPoolEvent(**base)


def test_freeze_authority_rejection():
    event = make_event(freeze_authority="BAD")
    cfg = {"freeze_blacklist": ["BAD"]}
    assert is_safe(event, cfg) is False


def test_dev_wallet_percentage_check():
    event = make_event()
    cfg = {"dev_share": 20.0, "max_dev_share": 10.0}
    assert is_safe(event, cfg) is False


def test_supply_sanity():
    event = make_event(liquidity=5.0)
    cfg = {"min_liquidity": 10.0}
    assert is_safe(event, cfg) is False
