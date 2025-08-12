import json
from pathlib import Path

import crypto_bot.utils.token_registry as registry


def test_overrides_have_source():
    data = json.loads(Path('crypto_bot/utils/token_overrides.json').read_text())
    for sym, info in data.items():
        if sym.startswith('_'):
            continue
        assert info.get('mint'), f'{sym} missing mint'
        assert info.get('source'), f'{sym} missing source'


def test_overrides_respect_flag(monkeypatch, tmp_path):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('enable_token_overrides: false\n')
    overrides = tmp_path / 'token_overrides.json'
    overrides.write_text(json.dumps({'AAA': {'mint': 'M', 'source': 'test'}}))
    monkeypatch.setattr(registry, 'CONFIG_FILE', cfg)
    monkeypatch.setattr(registry, 'OVERRIDES_FILE', overrides)
    assert registry._load_token_overrides() == {}
