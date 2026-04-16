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
    "max_position_pct": 0.90,
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
    "max_position_pct": 0.70,
    "vol_scale":        False,
    "stagger_exit":     True,
    "crash_brake_pct":  0.30,
}

# ── Risk Limits ────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "max_drawdown_halt":     0.50,   # halt trading if portfolio DD exceeds this
    "daily_stop_loss":       0.07,   # per-position daily stop-loss
    "alloc_drift_warn":      0.02,   # warn if actual allocation drifts >2% from target
    "alloc_drift_rebalance": 0.05,   # force rebalance if drift >5%
}

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
