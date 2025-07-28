import importlib.util
import pathlib
import pandas as pd

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

