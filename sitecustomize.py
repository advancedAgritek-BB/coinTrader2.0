import sys
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
