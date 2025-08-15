import sys
import types
from pathlib import Path

# Ensure local packages (crypto_bot, cointrainer) are importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:  # use real PyYAML if available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    if "yaml" not in sys.modules:
        sys.modules["yaml"] = types.SimpleNamespace(
            safe_load=lambda *a, **k: {},
            safe_dump=lambda *a, **k: "",
            dump=lambda *a, **k: "",
        )
