"""
overrides.py — validated, bounded runtime overrides for strategy parameters.
=============================================================================
Lets the Admin Panel adjust a WHITELISTED, BOUNDED set of parameters without
editing code — while the locked baseline in strategy_config.py stays intact.

Design (per the CLAUDE.md integrity guardrails):
  • The baseline lives in code and is never mutated on disk.
  • Overrides live in config/overrides.json (git-visible, reversible).
  • Only whitelisted keys within hard min/max bounds are accepted; anything
    else is rejected and logged. No free-form Python, ever.
  • Every applied override is logged to logs/config_audit.log.
  • strategy_config imports apply_overrides() at the end, so a saved override
    takes effect everywhere with no call-site changes. Delete the file (or use
    the panel's "reset to baseline") to restore the locked config exactly.

WHITELIST format:  "SECTION.key": (min, max, type)
  SECTION ∈ {A, B, REGIME, ALLOC, RISK, VOLTGT}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("config.overrides")

ROOT = Path(__file__).parent.parent
OVERRIDES_PATH = ROOT / "config" / "overrides.json"
AUDIT_LOG = ROOT / "logs" / "config_audit.log"

# ── Whitelist: key → (min, max, python_type) ────────────────────────────────
# Bounds are deliberately conservative — they cannot express a reckless config.
WHITELIST: dict[str, tuple] = {
    # Strategy A (aggressive)
    "A.ma_long":          (100, 300, int),
    "A.vix_exit":         (15.0, 40.0, float),
    "A.vix_reentry":      (10.0, 35.0, float),
    "A.max_position_pct": (0.30, 0.95, float),
    "A.crash_brake_pct":  (0.0, 0.50, float),
    # Strategy B (defensive)
    "B.ma_long":          (100, 300, int),
    "B.vix_exit":         (15.0, 40.0, float),
    "B.vix_reentry":      (10.0, 35.0, float),
    "B.max_position_pct": (0.20, 0.80, float),
    "B.crash_brake_pct":  (0.0, 0.50, float),
    # Regime detection
    "REGIME.ma_window":   (50, 250, int),
    "REGIME.vix_bull":    (10.0, 25.0, float),
    "REGIME.vix_hi_vol":  (18.0, 40.0, float),
    "REGIME.vix_smooth":  (1, 20, int),
    # Risk
    "RISK.max_drawdown_halt": (0.20, 0.60, float),
    "RISK.daily_stop_loss":   (0.03, 0.20, float),
    "RISK.alloc_drift_rebalance": (0.02, 0.15, float),
    # Vol-target overlay
    "VOLTGT.enabled":            (0, 1, int),      # 0/1 toggle
    "VOLTGT.target_annual_vol":  (0.30, 1.00, float),
    "VOLTGT.lookback":           (10, 60, int),
}


def _audit(msg: str) -> None:
    try:
        AUDIT_LOG.parent.mkdir(exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    except Exception:
        pass


def validate(key: str, value):
    """Return (ok, coerced_value_or_error). Rejects unknown/out-of-bounds keys."""
    if key not in WHITELIST:
        return False, f"'{key}' is not an overridable parameter"
    lo, hi, typ = WHITELIST[key]
    try:
        v = typ(value)
    except (TypeError, ValueError):
        return False, f"'{key}' must be {typ.__name__}"
    if not (lo <= v <= hi):
        return False, f"'{key}' must be in [{lo}, {hi}] (got {v})"
    return True, v


def load_raw() -> dict:
    if OVERRIDES_PATH.exists():
        try:
            return json.loads(OVERRIDES_PATH.read_text())
        except Exception as exc:
            logger.error(f"overrides.json unreadable ({exc}) — ignoring, using baseline")
    return {}


def save(overrides: dict) -> tuple[bool, str]:
    """Validate ALL keys, then persist. All-or-nothing so a bad key can't
    partially apply."""
    clean = {}
    for k, v in overrides.items():
        ok, res = validate(k, v)
        if not ok:
            return False, res
        clean[k] = res
    OVERRIDES_PATH.write_text(json.dumps(clean, indent=2))
    _audit(f"SAVE overrides: {clean}")
    return True, f"Saved {len(clean)} override(s)"


def clear() -> None:
    """Remove all overrides → exact locked baseline restored."""
    if OVERRIDES_PATH.exists():
        OVERRIDES_PATH.unlink()
    _audit("CLEAR overrides — reset to locked baseline")


_SECTION_TO_CONFIG = {
    "A": "STRATEGY_A_CONFIG", "B": "STRATEGY_B_CONFIG",
    "REGIME": "REGIME_CONFIG", "ALLOC": "ALLOC_CONFIG",
    "RISK": "RISK_CONFIG", "VOLTGT": "VOL_TARGET_CONFIG",
}


def apply_overrides(config_module) -> list[str]:
    """
    Merge validated overrides into the already-loaded config dicts on the given
    module (strategy_config). Returns a list of human-readable applied changes.
    Invalid entries are skipped and logged — never applied.
    """
    applied = []
    raw = load_raw()
    for key, value in raw.items():
        ok, res = validate(key, value)
        if not ok:
            logger.error(f"Skipping invalid override {key}={value}: {res}")
            continue
        section, _, param = key.partition(".")
        cfg_name = _SECTION_TO_CONFIG.get(section)
        if not cfg_name or not hasattr(config_module, cfg_name):
            continue
        cfg = getattr(config_module, cfg_name)
        if param == "enabled":
            res = bool(res)
        old = cfg.get(param)
        cfg[param] = res
        applied.append(f"{key}: {old} → {res}")
    if applied:
        logger.warning("Config overrides ACTIVE (baseline modified at runtime): "
                       + "; ".join(applied))
        _audit("APPLIED at import: " + "; ".join(applied))
    return applied
