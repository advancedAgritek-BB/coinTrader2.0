from crypto_bot.config import cfg
from crypto_bot import main


def test_apply_runtime_cfg_assigns_values():
    orig_strict = cfg.strict_cex
    orig_deny = list(cfg.denylist_symbols)
    try:
        config = {
            "mode": "cex",
            "strict_cex": False,
            "denylist_symbols": ["BAD/USDT"],
        }
        main._apply_runtime_cfg(config)
        assert cfg.strict_cex is False
        assert cfg.denylist_symbols == ["BAD/USDT"]
    finally:
        cfg.strict_cex = orig_strict
        cfg.denylist_symbols = orig_deny


def test_apply_runtime_cfg_defaults_to_cex():
    orig_strict = cfg.strict_cex
    orig_deny = list(cfg.denylist_symbols)
    try:
        cfg.strict_cex = False
        cfg.denylist_symbols = []
        main._apply_runtime_cfg({"mode": "cex"})
        assert cfg.strict_cex is True
        main._apply_runtime_cfg({"mode": "onchain"})
        assert cfg.strict_cex is False
    finally:
        cfg.strict_cex = orig_strict
        cfg.denylist_symbols = orig_deny
