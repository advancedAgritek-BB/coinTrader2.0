import ast
from pathlib import Path
from crypto_bot.utils.logger import setup_logger

SOURCE = Path('crypto_bot/main.py').read_text()
MODULE = ast.parse(SOURCE)
FUNC_SRC = ''
for node in MODULE.body:
    if isinstance(node, ast.FunctionDef) and node.name == '_emit_timing':
        FUNC_SRC = ast.get_source_segment(SOURCE, node)
        break


def test_emit_timing_logs(monkeypatch, tmp_path):
    log_file = tmp_path / 'bot.log'
    logger = setup_logger('timing_test', str(log_file))
    ns = {'log_cycle_metrics': lambda *a, **k: None, 'logger': logger, 'Path': Path}
    exec(FUNC_SRC, ns)
    ns['_emit_timing'](0.1, 0.2, 0.3, 0.6)
    text = log_file.read_text()
    assert 'Cycle timing' in text
    assert '0.10s' in text
    assert '0.60s' in text
