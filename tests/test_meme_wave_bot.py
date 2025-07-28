import importlib.util
import pathlib
import sys
import types
import pandas as pd

# The meme-wave strategy imports Solana helpers which depend on the optional
# ``solana`` package. Tests should run even when that dependency isn't
# installed, so we provide minimal stub modules mirroring the pattern used in
# ``tests/test_solana_executor.py``.
if importlib.util.find_spec("solana") is None:  # pragma: no cover - optional
    sys.modules.setdefault("solana", types.ModuleType("solana"))
    sys.modules.setdefault("solana.rpc", types.ModuleType("solana.rpc"))
    async_mod = types.ModuleType("solana.rpc.async_api")
    setattr(async_mod, "AsyncClient", object)
    sys.modules.setdefault("solana.rpc.async_api", async_mod)
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    sys.modules.setdefault("solana.transaction", types.ModuleType("solana.transaction"))
    sys.modules.setdefault("solana.rpc.api", types.ModuleType("solana.rpc.api"))

path = pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "strategy" / "meme_wave_bot.py"
spec = importlib.util.spec_from_file_location("meme_wave_bot", path)
meme_wave_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(meme_wave_bot)


def test_generate_signal_returns_tuple():
    df = pd.DataFrame({
        "open": [1, 2],
        "high": [1, 2],
        "low": [1, 2],
        "close": [1, 2],
        "volume": [1, 10],
    })
    score, direction = meme_wave_bot.generate_signal(df)
    assert isinstance(score, float)
    assert isinstance(direction, str)


def test_get_strategy_by_name():
    from crypto_bot import strategy_router

    fn = strategy_router.get_strategy_by_name("meme_wave_bot")
    assert callable(fn)

