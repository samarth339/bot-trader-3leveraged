"""Tests for config/overrides.py — the Admin Panel's validated overrides layer."""
import json
import importlib
import pytest


@pytest.fixture
def ov(tmp_path, monkeypatch):
    import config.overrides as O
    monkeypatch.setattr(O, "OVERRIDES_PATH", tmp_path / "overrides.json")
    monkeypatch.setattr(O, "AUDIT_LOG", tmp_path / "audit.log")
    return O


def test_validate_bounds_and_whitelist(ov):
    assert ov.validate("A.max_position_pct", 0.80)[0] is True
    assert ov.validate("A.max_position_pct", 0.99)[0] is False   # > 0.95
    assert ov.validate("A.max_position_pct", 0.10)[0] is False   # < 0.30
    assert ov.validate("A.not_a_param", 1)[0] is False           # unknown key
    assert ov.validate("RISK.max_drawdown_halt", 0.35)[0] is True


def test_save_is_all_or_nothing(ov):
    # one bad key ⇒ nothing saved
    ok, _ = ov.save({"A.max_position_pct": 0.80, "A.vix_exit": 999})
    assert ok is False
    assert not ov.OVERRIDES_PATH.exists()
    # all valid ⇒ saved
    ok, _ = ov.save({"A.max_position_pct": 0.80})
    assert ok is True
    assert json.loads(ov.OVERRIDES_PATH.read_text()) == {"A.max_position_pct": 0.80}


def test_apply_and_reset_roundtrip(ov, monkeypatch):
    import config.strategy_config as SC
    ov.save({"A.max_position_pct": 0.80, "VOLTGT.enabled": 1})
    # reload strategy_config with the patched overrides path
    monkeypatch.setattr("config.overrides.OVERRIDES_PATH", ov.OVERRIDES_PATH)
    importlib.reload(SC)
    assert SC.STRATEGY_A_CONFIG["max_position_pct"] == 0.80
    assert SC.VOL_TARGET_CONFIG["enabled"] is True
    # clear → baseline restored on reload
    ov.clear()
    importlib.reload(SC)
    assert SC.STRATEGY_A_CONFIG["max_position_pct"] == 0.85
    assert SC.VOL_TARGET_CONFIG["enabled"] is False


def test_baseline_untouched_when_no_overrides(ov):
    import importlib, config.strategy_config as SC
    importlib.reload(SC)
    assert SC.STRATEGY_A_CONFIG["max_position_pct"] == 0.85
    assert SC.STRATEGY_B_CONFIG["max_position_pct"] == 0.60


def teardown_module(module):
    # make sure the real strategy_config is back to baseline for other tests
    import importlib, config.strategy_config as SC
    importlib.reload(SC)
