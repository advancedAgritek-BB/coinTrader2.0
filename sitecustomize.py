import sys
from pathlib import Path
import types

try:  # use real PyYAML if available
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    if 'yaml' not in sys.modules:
        sys.modules['yaml'] = types.SimpleNamespace(
            safe_load=lambda *a, **k: {},
            safe_dump=lambda *a, **k: '',
            dump=lambda *a, **k: ''
        )

# Ensure ``src`` directory is importable for tests
src = Path(__file__).resolve().parent / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.append(str(src))

# Minimal stubs for optional third-party modules used in tests
sys.modules.setdefault("fakeredis", types.ModuleType("fakeredis"))
