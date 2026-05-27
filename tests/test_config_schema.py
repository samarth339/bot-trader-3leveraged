"""
test_config_schema.py — Configuration Contract Tests
======================================================
These tests are the contract between config/strategy_config.py and the rest
of the system.  Any change to a locked parameter will fail here *before* it
can silently corrupt backtests, signal generation, or live-order sizing.

They verify:
  - All required keys are present in each config section
  - Values are within safe operating ranges
  - Cross-config invariants hold (e.g. A is more aggressive than B)
  - Allocation weights sum to 1.0 for every regime
  - Locked post-expert-panel-v2 values haven't been accidentally changed
  - PORTFOLIO_DEFAULTS mirrors its source configs exactly

Run with:
    pytest tests/test_config_schema.py -v
"""

import pytest
from config.strategy_config import (
    REGIME_CONFIG,
    ALLOC_CONFIG,
    EXECUTION_CONFIG,
    STRATEGY_A_CONFIG,
    STRATEGY_B_CONFIG,
    RISK_CONFIG,
    PORTFOLIO_DEFAULTS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  REGIME_CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeConfig:
    """REGIME_CONFIG schema and value invariants."""

    REQUIRED_KEYS = {
        "ma_window", "vix_smooth", "vix_bull", "vix_hi_vol",
        "confirm_days", "t1_execution",
    }

    def test_all_required_keys_present(self):
        missing = self.REQUIRED_KEYS - REGIME_CONFIG.keys()
        assert not missing, f"REGIME_CONFIG missing keys: {missing}"

    def test_ma_window_is_positive_integer(self):
        w = REGIME_CONFIG["ma_window"]
        assert isinstance(w, int), f"ma_window must be int, got {type(w)}"
        assert w > 0, f"ma_window={w} must be positive"

    def test_vix_smooth_is_positive_integer(self):
        s = REGIME_CONFIG["vix_smooth"]
        assert isinstance(s, int), f"vix_smooth must be int, got {type(s)}"
        assert s > 0

    def test_vix_bull_strictly_below_hi_vol(self):
        """The 'uncertain' band [vix_bull, vix_hi_vol) must have positive width."""
        bull = REGIME_CONFIG["vix_bull"]
        hv   = REGIME_CONFIG["vix_hi_vol"]
        assert bull < hv, (
            f"vix_bull={bull} must be < vix_hi_vol={hv}; "
            "otherwise there is no 'uncertain' regime band"
        )

    def test_vix_thresholds_positive(self):
        assert REGIME_CONFIG["vix_bull"]   > 0
        assert REGIME_CONFIG["vix_hi_vol"] > 0

    def test_t1_execution_always_on(self):
        """Non-negotiable per CLAUDE.md — T-1 must always be True."""
        assert REGIME_CONFIG["t1_execution"] is True, (
            "T-1 execution MUST be enabled; see CLAUDE.md critical rule #1"
        )

    def test_confirm_days_non_negative(self):
        assert REGIME_CONFIG["confirm_days"] >= 0

    def test_locked_ma_window(self):
        """Locked at 130 after autoresearch (was 150 before). Regression guard."""
        assert REGIME_CONFIG["ma_window"] == 130, (
            "Regime MA window locked at 130 (autoresearch: was 150). "
            "Update this test if deliberately changing the window."
        )

    def test_locked_vix_smooth(self):
        """Locked at 5; confirm_days=2 was tested and made DD worse."""
        assert REGIME_CONFIG["vix_smooth"] == 5


# ══════════════════════════════════════════════════════════════════════════════
#  ALLOC_CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestAllocConfig:
    """Allocation weight contract for all three regimes."""

    REGIMES = ["bull", "uncertain", "high_vol"]

    def test_all_three_regimes_present(self):
        for r in self.REGIMES:
            assert r in ALLOC_CONFIG, f"Regime '{r}' missing from ALLOC_CONFIG"

    def test_weights_are_two_element_tuples(self):
        for regime, alloc in ALLOC_CONFIG.items():
            assert len(alloc) == 2, f"Regime '{regime}' must have exactly 2 weights"

    def test_weights_sum_to_one_for_each_regime(self):
        for regime, (wa, wb) in ALLOC_CONFIG.items():
            total = wa + wb
            assert abs(total - 1.0) < 1e-9, (
                f"Regime '{regime}': weight_a({wa}) + weight_b({wb}) = {total:.10f} ≠ 1.0"
            )

    def test_weights_are_non_negative(self):
        for regime, (wa, wb) in ALLOC_CONFIG.items():
            assert wa >= 0, f"weight_a for '{regime}' = {wa} is negative"
            assert wb >= 0, f"weight_b for '{regime}' = {wb} is negative"

    def test_bull_leans_strategy_a(self):
        """Bull regime must have more Strategy A (aggressive) than B."""
        wa, wb = ALLOC_CONFIG["bull"]
        assert wa > wb, f"Bull regime: weight_a({wa}) must > weight_b({wb})"

    def test_high_vol_leans_strategy_b(self):
        """High-vol must have more Strategy B (defensive) than A."""
        wa, wb = ALLOC_CONFIG["high_vol"]
        assert wb > wa, f"High-vol: weight_b({wb}) must > weight_a({wa})"

    def test_bull_more_aggressive_than_high_vol(self):
        """Bull Strategy-A weight must exceed high-vol Strategy-A weight."""
        assert ALLOC_CONFIG["bull"][0] > ALLOC_CONFIG["high_vol"][0]

    def test_locked_bull_allocation(self):
        """Locked at (0.9, 0.1) — at safe range maximum per autoresearch."""
        wa, wb = ALLOC_CONFIG["bull"]
        assert wa == pytest.approx(0.9), "Bull weight_a locked at 0.9"
        assert wb == pytest.approx(0.1), "Bull weight_b locked at 0.1"

    def test_locked_high_vol_allocation(self):
        """Locked at (0.25, 0.75)."""
        wa, wb = ALLOC_CONFIG["high_vol"]
        assert wa == pytest.approx(0.25)
        assert wb == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY_A_CONFIG  (BestCalmar — aggressive)
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyAConfig:
    """Strategy A (aggressive) contract."""

    REQUIRED_KEYS = {
        "name", "ma_long", "vix_exit", "vix_reentry", "confirm_bars",
        "max_position_pct", "vol_scale", "stagger_exit", "crash_brake_pct",
    }

    def test_all_required_keys_present(self):
        missing = self.REQUIRED_KEYS - STRATEGY_A_CONFIG.keys()
        assert not missing, f"STRATEGY_A_CONFIG missing keys: {missing}"

    def test_max_position_pct_in_valid_range(self):
        pct = STRATEGY_A_CONFIG["max_position_pct"]
        assert 0 < pct <= 1.0, f"max_position_pct={pct} outside (0, 1]"

    def test_vix_reentry_strictly_below_vix_exit(self):
        """Hysteresis: re-entry threshold must be lower to avoid whipsaws."""
        reentry = STRATEGY_A_CONFIG["vix_reentry"]
        exit_   = STRATEGY_A_CONFIG["vix_exit"]
        assert reentry < exit_, (
            f"vix_reentry={reentry} must be < vix_exit={exit_} (hysteresis required)"
        )

    def test_ma_long_positive(self):
        assert STRATEGY_A_CONFIG["ma_long"] > 0

    def test_confirm_bars_positive(self):
        assert STRATEGY_A_CONFIG["confirm_bars"] > 0

    def test_vol_scale_false(self):
        """Strategy A does NOT use vol_scale — only B does (expert panel v2)."""
        assert STRATEGY_A_CONFIG["vol_scale"] is False

    def test_stagger_exit_true(self):
        """Stagger exit must remain on for A — backtested as beneficial."""
        assert STRATEGY_A_CONFIG["stagger_exit"] is True

    def test_crash_brake_zero(self):
        """Strategy A has no crash brake — the aggressive strategy stays in."""
        assert STRATEGY_A_CONFIG["crash_brake_pct"] == 0.0

    def test_locked_max_position_pct(self):
        """Locked at 0.85 (expert panel v2, reduced from 0.95 for gap-risk buffer)."""
        assert STRATEGY_A_CONFIG["max_position_pct"] == pytest.approx(0.85), (
            "A max_position_pct locked at 0.85 (expert panel v2). "
            "Was 0.95, reduced for gap-risk cash buffer."
        )

    def test_locked_vix_parameters(self):
        """Locked after autoresearch: vix_exit=25, vix_reentry=24."""
        assert STRATEGY_A_CONFIG["vix_exit"]    == 25
        assert STRATEGY_A_CONFIG["vix_reentry"] == 24

    def test_locked_ma_long(self):
        """Locked at 190 (was 200 before autoresearch)."""
        assert STRATEGY_A_CONFIG["ma_long"] == 190


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY_B_CONFIG  (NearMiss — defensive)
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyBConfig:
    """Strategy B (defensive) contract."""

    REQUIRED_KEYS = {
        "name", "ma_long", "vix_exit", "vix_reentry", "confirm_bars",
        "max_position_pct", "vol_scale", "stagger_exit", "crash_brake_pct",
    }

    def test_all_required_keys_present(self):
        missing = self.REQUIRED_KEYS - STRATEGY_B_CONFIG.keys()
        assert not missing, f"STRATEGY_B_CONFIG missing keys: {missing}"

    def test_max_position_pct_in_valid_range(self):
        pct = STRATEGY_B_CONFIG["max_position_pct"]
        assert 0 < pct <= 1.0, f"max_position_pct={pct} outside (0, 1]"

    def test_vix_reentry_strictly_below_vix_exit(self):
        reentry = STRATEGY_B_CONFIG["vix_reentry"]
        exit_   = STRATEGY_B_CONFIG["vix_exit"]
        assert reentry < exit_, (
            f"vix_reentry={reentry} must be < vix_exit={exit_}"
        )

    def test_b_less_aggressive_than_a(self):
        """B is defensive — must have lower max position than A."""
        assert STRATEGY_B_CONFIG["max_position_pct"] < STRATEGY_A_CONFIG["max_position_pct"], (
            "Strategy B (defensive) must have lower max_position_pct than A (aggressive)"
        )

    def test_vol_scale_true(self):
        """Strategy B vol_scale=True is the expert panel v2 change — regression guard."""
        assert STRATEGY_B_CONFIG["vol_scale"] is True, (
            "Strategy B vol_scale must be True (expert panel v2). "
            "Was False before 2026-04-16."
        )

    def test_crash_brake_active(self):
        """B has crash brake; A does not. This is intentional."""
        assert STRATEGY_B_CONFIG["crash_brake_pct"] > 0.0, (
            "Strategy B crash_brake must be > 0 (defensive strategy)"
        )

    def test_locked_max_position_pct(self):
        """Locked at 0.60 (expert panel v2, reduced from 0.70)."""
        assert STRATEGY_B_CONFIG["max_position_pct"] == pytest.approx(0.60), (
            "B max_position_pct locked at 0.60 (expert panel v2). Was 0.70."
        )

    def test_locked_vix_parameters(self):
        """Locked after autoresearch: vix_exit=28, vix_reentry=22."""
        assert STRATEGY_B_CONFIG["vix_exit"]    == 28
        assert STRATEGY_B_CONFIG["vix_reentry"] == 22

    def test_locked_ma_long(self):
        """Locked at 150."""
        assert STRATEGY_B_CONFIG["ma_long"] == 150

    def test_locked_crash_brake_pct(self):
        """Crash brake at 30% (standalone crash_brake is lethal — see CLAUDE.md)."""
        assert STRATEGY_B_CONFIG["crash_brake_pct"] == pytest.approx(0.30)


# ══════════════════════════════════════════════════════════════════════════════
#  RISK_CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskConfig:
    """Risk limits contract."""

    REQUIRED_KEYS = {
        "max_drawdown_halt", "daily_stop_loss",
        "alloc_drift_warn", "alloc_drift_rebalance",
    }

    def test_all_required_keys_present(self):
        missing = self.REQUIRED_KEYS - RISK_CONFIG.keys()
        assert not missing, f"RISK_CONFIG missing keys: {missing}"

    def test_max_drawdown_halt_in_range(self):
        dd = RISK_CONFIG["max_drawdown_halt"]
        assert 0 < dd < 1.0, f"max_drawdown_halt={dd} should be in (0, 1)"

    def test_daily_stop_loss_in_range(self):
        sl = RISK_CONFIG["daily_stop_loss"]
        assert 0 < sl < 1.0, f"daily_stop_loss={sl} should be in (0, 1)"

    def test_drift_warn_strictly_below_rebalance(self):
        """Warning fires before rebalance — they must not be equal."""
        warn  = RISK_CONFIG["alloc_drift_warn"]
        rebal = RISK_CONFIG["alloc_drift_rebalance"]
        assert warn < rebal, (
            f"alloc_drift_warn={warn} must be < alloc_drift_rebalance={rebal}"
        )

    def test_drift_thresholds_positive(self):
        assert RISK_CONFIG["alloc_drift_warn"]      > 0
        assert RISK_CONFIG["alloc_drift_rebalance"] > 0

    def test_locked_drawdown_halt(self):
        """Tightened from 0.50 → 0.35 in expert panel v2.
        The OOS max DD was 55.4%, meaning the 50% halt would never have fired
        in the worst backtested scenario.  35% aligns with the 37.7% target.
        """
        assert RISK_CONFIG["max_drawdown_halt"] == pytest.approx(0.35), (
            "max_drawdown_halt locked at 0.35 (expert panel v2, was 0.50)"
        )

    def test_daily_stop_loss_value(self):
        assert RISK_CONFIG["daily_stop_loss"] == pytest.approx(0.07)

    def test_drift_rebalance_value(self):
        """5% drift gate — prevents unnecessary churn."""
        assert RISK_CONFIG["alloc_drift_rebalance"] == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTION_CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionConfig:
    """Execution model contract."""

    def test_model_is_close(self):
        """Production must use 'close' fill model — vwap breaks stop-loss logic."""
        assert EXECUTION_CONFIG["model"] == "close", (
            "Execution model MUST be 'close'. "
            "See config comments: vwap diverges from close-based stop checks."
        )

    def test_slippage_bps_positive(self):
        assert EXECUTION_CONFIG["slippage_bps"] > 0

    def test_slippage_bps_reasonable(self):
        """Round-trip slippage budget of 1–100 bps is realistic for MOC orders."""
        slippage = EXECUTION_CONFIG["slippage_bps"]
        assert 1 <= slippage <= 100, (
            f"slippage_bps={slippage} is outside the realistic 1–100 bps range"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO_DEFAULTS  (must mirror source configs)
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioDefaultsSync:
    """PORTFOLIO_DEFAULTS is a convenience flatmap — must stay in sync with sources."""

    def test_ma_window_mirrors_regime_config(self):
        assert PORTFOLIO_DEFAULTS["ma_window"] == REGIME_CONFIG["ma_window"]

    def test_vix_smooth_mirrors_regime_config(self):
        assert PORTFOLIO_DEFAULTS["vix_smooth"] == REGIME_CONFIG["vix_smooth"]

    def test_vix_bull_mirrors_regime_config(self):
        assert PORTFOLIO_DEFAULTS["vix_bull"] == REGIME_CONFIG["vix_bull"]

    def test_vix_hi_vol_mirrors_regime_config(self):
        assert PORTFOLIO_DEFAULTS["vix_hi_vol"] == REGIME_CONFIG["vix_hi_vol"]

    def test_alloc_bull_mirrors_alloc_config(self):
        assert PORTFOLIO_DEFAULTS["alloc_bull"] == ALLOC_CONFIG["bull"]

    def test_alloc_mid_mirrors_alloc_config(self):
        assert PORTFOLIO_DEFAULTS["alloc_mid"] == ALLOC_CONFIG["uncertain"]

    def test_alloc_hi_vol_mirrors_alloc_config(self):
        assert PORTFOLIO_DEFAULTS["alloc_hi_vol"] == ALLOC_CONFIG["high_vol"]

    def test_t1_flag_mirrors_regime_config(self):
        assert PORTFOLIO_DEFAULTS["t1"] == REGIME_CONFIG["t1_execution"]
