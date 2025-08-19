import asyncio
import importlib.util
import pathlib
import sys
import types

import pandas as pd

# create minimal package structure to load modules without heavy dependencies
pkg_root = types.ModuleType("crypto_bot")
sol_pkg = types.ModuleType("crypto_bot.solana")
utils_pkg = types.ModuleType("crypto_bot.utils")
pkg_root.volatility_filter = types.ModuleType("crypto_bot.volatility_filter")
pkg_root.solana = sol_pkg
pkg_root.utils = utils_pkg
pkg_root.volatility_filter.calc_atr = lambda *_a, **_k: pd.Series([0.0])
import importlib.machinery
pkg_root.__spec__ = importlib.machinery.ModuleSpec("crypto_bot", None, is_package=True)
pkg_root.__spec__.submodule_search_locations = []
pkg_root.__path__ = []
sol_pkg.__spec__ = importlib.machinery.ModuleSpec("crypto_bot.solana", None, is_package=True)
sol_pkg.__spec__.submodule_search_locations = [str(pathlib.Path("crypto_bot/solana"))]
sol_pkg.__path__ = [str(pathlib.Path("crypto_bot/solana"))]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec("crypto_bot.utils", None, is_package=True)
utils_pkg.__spec__.submodule_search_locations = [str(pathlib.Path("crypto_bot/utils"))]
utils_pkg.__path__ = [str(pathlib.Path("crypto_bot/utils"))]

sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.solana", sol_pkg)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)
sys.modules.setdefault("crypto_bot.volatility_filter", pkg_root.volatility_filter)

# load watcher and solana_scanner modules directly
watcher_spec = importlib.util.spec_from_file_location(
    "crypto_bot.solana.watcher",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "solana" / "watcher.py",
)
watcher = importlib.util.module_from_spec(watcher_spec)
sys.modules["crypto_bot.solana.watcher"] = watcher
watcher_spec.loader.exec_module(watcher)

scanner_spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.solana_scanner",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "solana_scanner.py",
)
solana_scanner = importlib.util.module_from_spec(scanner_spec)
sys.modules["crypto_bot.utils.solana_scanner"] = solana_scanner
scanner_spec.loader.exec_module(solana_scanner)

scanner2_spec = importlib.util.spec_from_file_location(
    "crypto_bot.solana.scanner",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "solana" / "scanner.py",
)
solana_scan_mod = importlib.util.module_from_spec(scanner2_spec)
sys.modules["crypto_bot.solana.scanner"] = solana_scan_mod
scanner2_spec.loader.exec_module(solana_scan_mod)


# Use PoolWatcher and NewPoolEvent from the loaded watcher module
PoolWatcher = watcher.PoolWatcher
NewPoolEvent = watcher.NewPoolEvent


# sample events yielded by the mocked watcher
events = [
    NewPoolEvent("P1", "A", "C1", 80.0, 3),
    NewPoolEvent("P2", "B", "C1", 40.0, 5),
    NewPoolEvent("P3", "C", "C1", 70.0, 1),
    NewPoolEvent("P4", "D", "C1", 70.0, 7),
    NewPoolEvent("P5", "E", "C1", 90.0, 4),
]


async def watch_stub(self):
    for evt in events:
        yield evt


def test_get_solana_new_tokens_filters_and_limits(monkeypatch):
    monkeypatch.setattr(PoolWatcher, "watch", watch_stub)
    monkeypatch.setenv("HELIUS_KEY", "k")
    monkeypatch.setattr(PoolWatcher, "setup_webhook", lambda self, k: None)

    cfg = {"min_liquidity": 0}
    tokens = asyncio.run(solana_scan_mod.get_solana_new_tokens(cfg))
    assert tokens == ["A", "B", "C", "D", "E"]


async def watch_simple(self):
    for evt in events:
        yield evt


def test_scanner_collects_tokens(monkeypatch):
    monkeypatch.setattr(PoolWatcher, "watch", watch_simple)
    monkeypatch.setattr(solana_scan_mod, "TOKEN_MINTS", {}, raising=False)
    monkeypatch.setenv("HELIUS_KEY", "k")
    monkeypatch.setattr(PoolWatcher, "setup_webhook", lambda self, k: None)

    cfg = {
        "min_liquidity": 0,
        "min_tx_count": 0,
    }

    tokens = asyncio.run(solana_scan_mod.get_solana_new_tokens(cfg))
    assert tokens == ["A", "B", "C", "D", "E"]
