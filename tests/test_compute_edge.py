import pytest
pytest.importorskip("pandas")
import pandas as pd
import pytest
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]

crypto_bot = types.ModuleType("crypto_bot")
utils_pkg = types.ModuleType("crypto_bot.utils")
sys.modules.setdefault("crypto_bot", crypto_bot)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)

for mod_name in ["logger", "strategy_utils"]:
    path = ROOT / f"crypto_bot/utils/{mod_name}.py"
    spec = importlib.util.spec_from_file_location(f"crypto_bot.utils.{mod_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"crypto_bot.utils.{mod_name}"] = module
    spec.loader.exec_module(module)

compute_edge = sys.modules["crypto_bot.utils.strategy_utils"].compute_edge


def test_compute_edge_basic(tmp_path):
    file = tmp_path / "pnl.csv"
    data = [
        {"strategy": "scalp", "pnl": 10},
        {"strategy": "scalp", "pnl": -5},
        {"strategy": "scalp", "pnl": 20},
        {"strategy": "scalp", "pnl": -10},
        {"strategy": "other", "pnl": 1},
    ]
    pd.DataFrame(data).to_csv(file, index=False)
    edge = compute_edge("scalp", drawdown_penalty=0.05, path=file)
    assert edge == pytest.approx(0.5)
