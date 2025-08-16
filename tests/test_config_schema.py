import pytest
from pydantic import ValidationError

from schema.scanner import (
    ScannerConfig,
    SolanaScannerConfig,
    SolanaScannerApiKeys,
    PythConfig,
)


def test_symbols_required_when_not_scanning():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=False, symbols=[])


def test_invalid_market_type():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=True, exchange_market_types=["invalid"])


def test_positive_values_enforced():
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=True, scan_lookback_limit=0)
    with pytest.raises(ValidationError):
        ScannerConfig(scan_markets=True, cycle_lookback_limit=0)


def test_solana_scanner_defaults():
    cfg = SolanaScannerConfig()
    assert cfg.enabled is True
    assert cfg.interval_minutes == 0.1
    assert cfg.max_tokens_per_scan == 200
    assert cfg.min_volume_usd == 10.0
    assert cfg.gecko_search is True
    assert cfg.api_keys is None


def test_solana_scanner_invalid_type():
    with pytest.raises(ValidationError):
        SolanaScannerConfig(interval_minutes="x")


def test_solana_scanner_env_override(monkeypatch):
    monkeypatch.setenv("MORALIS_KEY", "env_key")
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "env_lunar")
    cfg = SolanaScannerConfig(api_keys=SolanaScannerApiKeys())
    assert cfg.api_keys.moralis == "env_key"
    assert cfg.api_keys.lunarcrush_api_key == "env_lunar"


def test_pyth_defaults():
    cfg = PythConfig()
    assert cfg.enabled is False
    assert cfg.solana_endpoint == ""
    assert cfg.solana_ws_endpoint == ""
    assert cfg.program_id == ""


def test_pyth_invalid_type():
    with pytest.raises(ValidationError):
        PythConfig(extra_field=1)

