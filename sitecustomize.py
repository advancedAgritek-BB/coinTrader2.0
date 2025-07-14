import sys, types

# Provide a minimal yaml stub before tests can monkeypatch
if 'yaml' not in sys.modules:
    sys.modules['yaml'] = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: ''
    )
