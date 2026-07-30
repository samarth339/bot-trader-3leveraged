"""
Strategy Configuration — Single Source of Truth
================================================
ALL parameters that affect signals, regimes, allocations, or execution
live here. Changing a parameter in this file propagates everywhere.

DO NOT hard-code any of these values elsewhere.
"""

# ── Regime Classification ──────────────────────────────────────────────────
REGIME_CONFIG = {
    "ma_window":      130,    # SMA window for bull/bear detection
    "vix_smooth":     5,      # VIX rolling average (reduces noise)
    "vix_bull":       18.0,   # VIX below this = calm (bull regime)
    "vix_hi_vol":     25.0,   # VIX at/above this = danger (high-vol regime)
    "confirm_days":   1,      # consecutive days before regime can change
    "t1_execution":   True,   # ALWAYS use previous-day signals for today's action
}

# ── Portfolio Allocations  (strategy_A_weight, strategy_B_weight) ─────────
ALLOC_CONFIG = {
    "bull":     (0.9, 0.1),   # strong bull:  lean into aggressive strategy
    "uncertain": (0.65, 0.35),  # uncertain:    balanced
    "high_vol": (0.25, 0.75),   # high vol:     lean into defensive strategy
}

# ── Execution Model ────────────────────────────────────────────────────────
EXECUTION_CONFIG = {
    # "close"     → fill at same-bar close (legacy, not recommended)
    # "vwap"      → fill at (O+H+L+C)/4 proxy (default — realistic intraday avg)
    # "next_open" → buffer signal, fill at next bar's open (most conservative)
    # "close"     → PRODUCTION DEFAULT. Strategies validated with this.
    #               T-1 regime guard (not execution) is the primary protection.
    # "vwap"      → (O+H+L+C)/4 intraday proxy. BREAKS stop-losses: entry at VWAP
    #               diverges from close-based daily-stop checks — not safe to use
    #               unless stop-loss logic is rewritten to use entry-price-relative checks.
    # "next_open" → Most conservative but incompatible with crash-brake strategies
    #               (1-bar lag between signal and execution lets crash brakes re-trigger).
    "model":        "close",
    "slippage_bps":  10,        # round-trip slippage budget per fill
}

# ── Strategy A: Best-Calmar (aggressive) ──────────────────────────────────
STRATEGY_A_CONFIG = {
    "name":             "BestCalmar",
    "ma_long":          190,
    "vix_exit":         25,
    "vix_reentry":      24,
    "confirm_bars":     2,
    "max_position_pct": 0.85,   # reduced from 0.95 — gap-risk buffer (expert panel v2)
    "vol_scale":        False,
    "stagger_exit":     True,
    "crash_brake_pct":  0.0,
}

# ── Strategy B: Near-Miss (defensive) ─────────────────────────────────────
STRATEGY_B_CONFIG = {
    "name":             "NearMiss",
    "ma_long":          150,
    "vix_exit":         28,
    "vix_reentry":      22,
    "confirm_bars":     4,
    "max_position_pct": 0.60,   # reduced from 0.70 — gap-risk buffer (expert panel v2)
    "vol_scale":        True,   # enabled — gradual VIX-tier de-risking (expert panel v2)
    "stagger_exit":     True,
    "crash_brake_pct":  0.30,
}

# ── Risk Limits ────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "max_drawdown_halt":     0.35,   # halt trading if portfolio DD exceeds this
                                     # tightened from 0.50 — OOS max DD was 55.4%,
                                     # meaning 50% halt never would have fired in worst
                                     # backtested scenario. 35% aligns with 37.7% full-
                                     # period max DD target (expert panel v2).
    "daily_stop_loss":       0.07,   # per-position daily stop-loss
    "alloc_drift_warn":      0.02,   # warn if actual allocation drifts >2% from target
    "alloc_drift_rebalance": 0.05,   # force rebalance if drift >5%
}

# ── Volatility-Targeted Exposure Overlay ───────────────────────────────────
# Portfolio-level exposure scalar applied ON TOP of the regime/exposure system.
# Scales the blended TQQQ target by (target_vol / realized_3x_vol), so exposure
# falls BEFORE VIX-tier thresholds trigger when realized volatility rises, and
# rises toward full in calm tapes. Rationale: volatility is far more
# autocorrelated/forecastable than returns, and on a 3x product cutting
# exposure in high-vol regimes reduces both drawdown AND decay drag.
#
# T-1 safe: realized vol is computed from QQQ closes shifted by 1 bar.
# DEFAULT OFF — this is an experimental overlay, not part of the locked
# baseline. Promote to enabled ONLY after an honest in- + out-of-sample
# backtest shows it improves risk-adjusted return, and with owner sign-off.
VOL_TARGET_CONFIG = {
    "enabled":           True,    # ENABLED 2026-07-29 (owner-authorized) at 55%
                                  # after the baseline test window. Backtest A/B
                                  # showed Calmar↑ and MaxDD↓ in all periods incl.
                                  # OOS (34.8→30.4%). Live from the 2026-07-29 reset.
    "target_annual_vol": 0.55,    # target annualized portfolio vol (55%)
    "lookback":          20,      # trading days for realized-vol estimate
    "leverage":          3.0,     # TQQQ ≈ 3x QQQ (realized-vol proxy multiplier)
    "floor":             0.0,     # min exposure scalar
    "cap":               1.0,     # max scalar (never lever ABOVE the blended target)
}

# ── Apply validated runtime overrides (Admin Panel) ────────────────────────
# The baseline above stays in code and is never mutated on disk. Any overrides
# saved via the Admin Panel (config/overrides.json, whitelisted + bounded) are
# merged into the dicts here — BEFORE PORTFOLIO_DEFAULTS is built — so they take
# effect everywhere. Delete overrides.json (or use the panel's reset) to restore
# the locked baseline exactly. Applied changes are logged to logs/config_audit.log.
try:
    import sys as _sys
    from config.overrides import apply_overrides as _apply_overrides
    _apply_overrides(_sys.modules[__name__])
except Exception as _ovr_exc:   # never let overrides break config import
    import logging as _logging
    _logging.getLogger("config.strategy_config").error(
        f"Config override application skipped: {_ovr_exc}")

# ── Convenience: flattened dict for DualPortfolioBacktester constructor ────
PORTFOLIO_DEFAULTS = dict(
    ma_window    = REGIME_CONFIG["ma_window"],
    vix_smooth   = REGIME_CONFIG["vix_smooth"],
    vix_bull     = REGIME_CONFIG["vix_bull"],
    vix_hi_vol   = REGIME_CONFIG["vix_hi_vol"],
    confirm_days = REGIME_CONFIG["confirm_days"],
    t1           = REGIME_CONFIG["t1_execution"],
    alloc_bull   = ALLOC_CONFIG["bull"],
    alloc_mid    = ALLOC_CONFIG["uncertain"],
    alloc_hi_vol = ALLOC_CONFIG["high_vol"],
)
