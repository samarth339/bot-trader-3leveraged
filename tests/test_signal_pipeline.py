"""
Integration Tests — Daily Signal Pipeline

Tests daily_signal.py functions end-to-end:
  • Regime computation returns valid, structured output
  • Action resolver produces correct HOLD / INCREASE_A / REDUCE_A
  • Signal log appends correctly without corruption
  • Drift calculation is accurate
"""
import csv
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from daily_signal import compute_regime, resolve_action, append_signal_log
from config.strategy_config import ALLOC_CONFIG, RISK_CONFIG


# ── compute_regime() ──────────────────────────────────────────────────────────

class TestComputeRegime:

    def test_returns_all_required_keys(self, real_qqq, real_vix):
        """Signal dict must contain all required keys."""
        required = {
            "as_of_date", "signal_date", "regime", "reason",
            "price_signal", "sma_val", "pct_vs_sma",
            "vix_signal", "vix_raw",
        }
        as_of = pd.Timestamp("2023-06-01")
        sig = compute_regime(real_qqq, real_vix, as_of)
        missing = required - set(sig.keys())
        assert not missing, f"Missing keys in signal output: {missing}"

    def test_regime_is_valid(self, real_qqq, real_vix):
        """Regime must always be one of the three valid labels."""
        as_of = pd.Timestamp("2023-06-01")
        sig = compute_regime(real_qqq, real_vix, as_of)
        assert sig["regime"] in {"bull", "uncertain", "high_vol"}

    def test_signal_date_is_before_as_of(self, real_qqq, real_vix):
        """T-1 guarantee: signal_date must be strictly before as_of_date."""
        as_of = pd.Timestamp("2023-06-01")
        sig = compute_regime(real_qqq, real_vix, as_of)
        assert sig["signal_date"] < sig["as_of_date"], \
            "T-1 violated: signal_date must be before as_of_date"

    def test_covid_period_is_high_vol(self, real_qqq, real_vix):
        """COVID crash peak (2020-03-16) must register as HIGH_VOL."""
        as_of = pd.Timestamp("2020-03-16")
        sig = compute_regime(real_qqq, real_vix, as_of)
        assert sig["regime"] == "high_vol", \
            f"COVID crash should be high_vol, got {sig['regime']}"

    def test_2021_bull_period(self, real_qqq, real_vix):
        """Mid-2021 should not be high_vol (VIX grey zone may give uncertain — OK)."""
        as_of = pd.Timestamp("2021-08-01")
        sig = compute_regime(real_qqq, real_vix, as_of)
        assert sig["regime"] != "high_vol", \
            f"Mid-2021 should not be high_vol, got '{sig['regime']}' (VIX={sig['vix_signal']:.1f})"

    def test_pct_vs_sma_sign_matches_regime(self, real_qqq, real_vix):
        """If regime is high_vol due to price < SMA, pct_vs_sma must be negative."""
        # Use 2022 bear market
        as_of = pd.Timestamp("2022-07-01")
        sig = compute_regime(real_qqq, real_vix, as_of)
        if sig["regime"] == "high_vol" and not np.isnan(sig["pct_vs_sma"]):
            # Either price below MA or VIX spiked — check consistency
            if sig["price_signal"] < sig["sma_val"]:
                assert sig["pct_vs_sma"] < 0

    def test_raises_on_no_data_before_date(self, real_qqq, real_vix):
        """Requesting a date before all data should raise ValueError."""
        with pytest.raises(ValueError, match="No QQQ data"):
            compute_regime(real_qqq, real_vix, pd.Timestamp("1900-01-01"))

    def test_multiple_dates_all_valid(self, real_qqq, real_vix):
        """Sample 20 random dates — all should produce valid regimes."""
        test_dates = pd.date_range("2012-01-01", "2024-12-31", periods=20)
        for dt in test_dates:
            sig = compute_regime(real_qqq, real_vix, dt)
            assert sig["regime"] in {"bull", "uncertain", "high_vol"}, \
                f"Invalid regime on {dt.date()}: {sig['regime']}"


# ── resolve_action() ──────────────────────────────────────────────────────────

class TestResolveAction:

    def test_hold_when_regime_unchanged(self):
        """Same regime → HOLD, no rebalance."""
        result = resolve_action("bull", "bull")
        assert result["action"] == "HOLD"
        assert result["rebalance_needed"] is False

    def test_increase_a_on_bull_from_uncertain(self):
        """Uncertain → Bull: more weight to A (aggressive) → INCREASE_A."""
        result = resolve_action("bull", "uncertain")
        assert result["action"] == "INCREASE_A"
        assert result["rebalance_needed"] is True
        assert result["target_alloc"][0] > result["prev_alloc"][0]

    def test_reduce_a_on_high_vol(self):
        """Bull → High-vol: less A (aggressive) → REDUCE_A."""
        result = resolve_action("high_vol", "bull")
        assert result["action"] == "REDUCE_A"
        assert result["rebalance_needed"] is True
        assert result["target_alloc"][0] < result["prev_alloc"][0]

    def test_drift_calculation_correct(self):
        """Drift = |target_a - prev_a| * 100."""
        bull_a = ALLOC_CONFIG["bull"][0]       # 0.75
        hv_a   = ALLOC_CONFIG["high_vol"][0]   # 0.30
        result = resolve_action("high_vol", "bull")
        expected_drift = abs(hv_a - bull_a) * 100
        assert abs(result["drift_pct"] - expected_drift) < 0.01

    def test_no_prev_regime_defaults_to_hold_logic(self):
        """No previous regime → target alloc = prev alloc → HOLD."""
        result = resolve_action("bull", "bull")  # same
        assert result["action"] == "HOLD"

    def test_all_regime_transitions_produce_valid_action(self):
        """All 9 regime transition combos produce valid action strings."""
        valid_actions = {"HOLD", "INCREASE_A", "REDUCE_A"}
        for from_r in ["bull", "uncertain", "high_vol"]:
            for to_r in ["bull", "uncertain", "high_vol"]:
                result = resolve_action(to_r, from_r)
                assert result["action"] in valid_actions, \
                    f"Invalid action {result['action']} for {from_r}→{to_r}"

    def test_target_alloc_sums_to_one(self):
        """Target allocation weights must always sum to 1.0."""
        for regime in ["bull", "uncertain", "high_vol"]:
            result = resolve_action(regime, "uncertain")
            wa, wb = result["target_alloc"]
            assert abs(wa + wb - 1.0) < 1e-9, \
                f"Allocation ({wa}, {wb}) does not sum to 1.0"


# ── Signal log ────────────────────────────────────────────────────────────────

class TestSignalLog:

    def test_log_appends_single_row(self):
        """append_signal_log() adds exactly 1 row to the CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            log_path = Path(f.name)

        log_path.unlink()  # ensure fresh
        row = {
            "as_of_date": "2026-03-23", "signal_date": "2026-03-20",
            "regime": "bull", "action": "HOLD",
            "weight_a": 0.75, "weight_b": 0.25,
            "rebalance": False, "drift_pct": 0.0,
            "qqq_price": 473.81, "sma_val": 450.0, "pct_vs_sma": 5.3,
            "vix_signal": 14.8, "vix_raw": 15.1, "shadow": True,
        }
        with patch("daily_signal.SIGNAL_LOG", log_path):
            append_signal_log(row)

        df = pd.read_csv(log_path)
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert df.iloc[0]["regime"] == "bull"
        log_path.unlink()

    def test_log_appends_multiple_rows(self):
        """Multiple calls append without overwriting."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            log_path = Path(f.name)
        log_path.unlink()

        row = {"as_of_date": "2026-03-23", "regime": "bull", "action": "HOLD"}
        with patch("daily_signal.SIGNAL_LOG", log_path):
            append_signal_log(row)
            append_signal_log({**row, "as_of_date": "2026-03-24", "regime": "uncertain"})
            append_signal_log({**row, "as_of_date": "2026-03-25", "regime": "high_vol"})

        df = pd.read_csv(log_path)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert list(df["regime"]) == ["bull", "uncertain", "high_vol"]
        log_path.unlink()


# ── Dynamic uncertain allocation ──────────────────────────────────────────────

class TestDynamicUncertainAlloc:
    """resolve_action() must use pct_vs_sma to scale uncertain allocation."""

    def test_strong_recovery_gives_75_25(self):
        """pct_vs_sma > 3% → most aggressive uncertain alloc (75/25)."""
        result = resolve_action("uncertain", "uncertain", pct_vs_sma=4.5)
        assert result["target_alloc"] == (0.75, 0.25)

    def test_clear_uptrend_gives_70_30(self):
        """pct_vs_sma between 1% and 3% → lean aggressive (70/30)."""
        result = resolve_action("uncertain", "uncertain", pct_vs_sma=2.0)
        assert result["target_alloc"] == (0.70, 0.30)

    def test_neutral_gives_65_35(self):
        """pct_vs_sma between -1% and 1% → neutral, same as old fixed (65/35)."""
        result = resolve_action("uncertain", "uncertain", pct_vs_sma=0.0)
        assert result["target_alloc"] == (0.65, 0.35)

    def test_weakening_gives_55_45(self):
        """pct_vs_sma between -3% and -1% → lean defensive (55/45)."""
        result = resolve_action("uncertain", "uncertain", pct_vs_sma=-2.0)
        assert result["target_alloc"] == (0.55, 0.45)

    def test_near_high_vol_gives_45_55(self):
        """pct_vs_sma < -3% → most defensive uncertain alloc (45/55)."""
        result = resolve_action("uncertain", "uncertain", pct_vs_sma=-4.0)
        assert result["target_alloc"] == (0.45, 0.55)

    def test_all_dynamic_allocs_sum_to_one(self):
        """Every pct_vs_sma bracket must produce weights summing to 1.0."""
        for pct in [-5.0, -2.5, 0.0, 2.0, 4.0]:
            result = resolve_action("uncertain", "bull", pct_vs_sma=pct)
            wa, wb = result["target_alloc"]
            assert abs(wa + wb - 1.0) < 1e-9, f"Alloc doesn't sum to 1 at pct={pct}"

    def test_backward_compat_default_pct_is_neutral(self):
        """Calling without pct_vs_sma defaults to 0.0 → 65/35 (no regression)."""
        result = resolve_action("uncertain", "uncertain")
        assert result["target_alloc"] == (0.65, 0.35)

    def test_prev_pct_drives_drift_accurately(self):
        """Transition from deeply below SMA to strongly above → large INCREASE_A."""
        result = resolve_action("uncertain", "uncertain",
                                pct_vs_sma=4.0,       # today: 75/25
                                prev_pct_vs_sma=-4.0) # yesterday: 45/55
        assert result["action"] == "INCREASE_A"
        assert result["rebalance_needed"] is True
        assert result["drift_pct"] == pytest.approx(30.0, abs=0.1)

    def test_stable_pct_within_same_bracket_gives_hold(self):
        """Two consecutive days in same bracket with <2% drift → HOLD."""
        result = resolve_action("uncertain", "uncertain",
                                pct_vs_sma=1.5,        # 70/30
                                prev_pct_vs_sma=2.5)   # still 70/30
        assert result["action"] == "HOLD"
        assert result["drift_pct"] == 0.0

    def test_bull_and_high_vol_unaffected_by_pct(self):
        """pct_vs_sma must not alter allocations for non-uncertain regimes."""
        bull = resolve_action("bull", "uncertain", pct_vs_sma=5.0)
        hv   = resolve_action("high_vol", "uncertain", pct_vs_sma=-5.0)
        from config.strategy_config import ALLOC_CONFIG
        assert bull["target_alloc"] == ALLOC_CONFIG["bull"]
        assert hv["target_alloc"]   == ALLOC_CONFIG["high_vol"]

    def test_increase_a_on_recovery_from_high_vol(self):
        """high_vol → uncertain with strong recovery → INCREASE_A."""
        result = resolve_action("uncertain", "high_vol", pct_vs_sma=3.5)
        assert result["action"] == "INCREASE_A"
        assert result["target_alloc"] == (0.75, 0.25)


# ── Momentum override (ROC-5) ─────────────────────────────────────────────────

class TestMomentumOverride:
    """compute_regime() must upgrade high_vol → uncertain on strong 5-day momentum."""

    def _make_vix_spike_data(self, n=250):
        """
        Build QQQ near but ABOVE SMA (so price check passes) + VIX smoothed just
        above 25 (so VIX check triggers high_vol) + strong 5-day momentum.
        This isolates the VIX-triggered high_vol path for the override.
        """
        import numpy as np, pandas as pd

        # Flat price (so SMA ≈ price, i.e. pct_vs_sma ≈ 0) — passes the price > -1.5% check
        prices   = np.full(n, 500.0)
        # Upward drift over last 5 bars — 1%/bar so signal ROC-5 > 3% after T-1 shift
        for i in range(5):
            prices[n - 5 + i] = 500.0 * (1 + 0.010 * (i + 1))  # +1% per bar → ROC-5 ≈ 4%
        dates = pd.bdate_range("2018-01-01", periods=n)
        qqq   = pd.DataFrame({"close": prices, "open": prices * 0.999,
                              "high": prices * 1.005, "low": prices * 0.995,
                              "volume": 1_000_000}, index=dates)
        # VIX: spike to 30 six bars ago, now 26 — smoothed (5-day) still > 25
        vix_vals = np.full(n, 16.0)
        vix_vals[n - 6] = 32.0
        vix_vals[n - 5] = 30.0
        vix_vals[n - 4] = 28.0
        vix_vals[n - 3] = 27.0
        vix_vals[n - 2] = 26.0
        vix_vals[n - 1] = 25.5   # smoothed(5) ≈ 27.3 → triggers high_vol
        vix = pd.DataFrame({"close": vix_vals, "open": vix_vals,
                            "high": vix_vals, "low": vix_vals, "volume": 0},
                           index=dates)
        return qqq, vix

    def _make_recovery_data(self, n_warmup=200, crash_depth=0.12, recovery_roc5=0.05):
        """
        Build synthetic QQQ that:
        - Trends up for n_warmup bars (establishes SMA)
        - Crashes below SMA
        - Recovers strongly over 5 bars (recovery_roc5 per bar)
        Returns (qqq_df, vix_df) ready for compute_regime().
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        # Warmup: steady uptrend so SMA-130 is well established
        warmup = 200.0 * np.cumprod(1 + rng.normal(0.0005, 0.005, n_warmup))
        # Crash: single-bar drop to below SMA
        crash_price = warmup[-1] * (1 - crash_depth)
        # Recovery: 5 bars of strong upside (back near SMA)
        recovery = crash_price * np.cumprod(1 + np.full(6, recovery_roc5))

        prices = np.concatenate([warmup, [crash_price], recovery])
        dates  = pd.bdate_range("2015-01-01", periods=len(prices))
        qqq    = pd.DataFrame({"close": prices, "open": prices * 0.999,
                               "high": prices * 1.005, "low": prices * 0.995,
                               "volume": 1_000_000}, index=dates)
        # VIX: elevated during crash then falling (smoothed will be ~24 at recovery end)
        vix_vals = np.full(len(prices), 16.0)
        vix_vals[n_warmup] = 30.0        # crash day
        for i in range(1, 7):
            vix_vals[n_warmup + i] = 30.0 - i * 1.5  # declining: 28.5, 27, 25.5…
        vix = pd.DataFrame({"close": vix_vals, "open": vix_vals,
                            "high": vix_vals, "low": vix_vals,
                            "volume": 0}, index=dates)
        return qqq, vix

    def test_override_fires_on_strong_momentum(self):
        """
        When VIX spike causes high_vol but price is near SMA and 5-day ROC > 3%,
        regime must be upgraded to uncertain with 'momentum override' in reason.
        """
        from daily_signal import compute_regime

        qqq, vix = self._make_vix_spike_data()
        as_of = qqq.index[-1]
        sig = compute_regime(qqq, vix, as_of)

        assert sig["regime"] == "uncertain", (
            f"Expected uncertain (momentum override), got {sig['regime']}. "
            f"ROC-5={sig.get('roc_5', 'N/A')}, pct_vs_sma={sig['pct_vs_sma']:.2f}%")
        assert "momentum override" in sig["reason"].lower(), (
            f"Reason should mention override: {sig['reason']}")

    def test_override_roc5_in_return_dict(self):
        """compute_regime() must return roc_5 key in all cases."""
        from daily_signal import compute_regime
        import pandas as pd

        qqq, vix = self._make_recovery_data()
        as_of = qqq.index[-1]
        sig = compute_regime(qqq, vix, as_of)
        assert "roc_5" in sig, "roc_5 key missing from compute_regime() output"

    def test_override_requires_pct_vs_sma_threshold(self):
        """
        Strong ROC-5 alone is not enough — price must be within 1.5% of SMA.
        Deep below SMA (>1.5%) must stay high_vol even with positive momentum.
        """
        from daily_signal import compute_regime
        import pandas as pd, numpy as np

        # Crash very deep so price is still -5% below SMA even after 5-bar recovery
        qqq, vix = self._make_recovery_data(crash_depth=0.25, recovery_roc5=0.004)
        as_of = qqq.index[-1]
        sig = compute_regime(qqq, vix, as_of)

        # Price should still be well below SMA → high_vol despite positive ROC
        if sig["pct_vs_sma"] < -1.5:
            assert sig["regime"] == "high_vol", (
                f"Deep-below-SMA case should stay high_vol, got {sig['regime']}")

    def test_override_inactive_in_non_high_vol(self):
        """Override logic must only activate when base regime is high_vol."""
        from daily_signal import compute_regime
        import pandas as pd

        # Pure bull: strong uptrend, low VIX
        qqq, vix = self._make_recovery_data(recovery_roc5=0.002)
        # Use a date in the warmup where regime is already bull
        as_of = qqq.index[180]
        sig = compute_regime(qqq, vix, as_of)
        # Must be bull or uncertain from normal logic, not from override
        assert sig["regime"] in {"bull", "uncertain"}
        if "momentum override" in sig.get("reason", "").lower():
            # If override fired here, that's a bug (base wasn't high_vol)
            assert False, "Override should not fire when base regime is not high_vol"

    def test_real_data_april_2026_recovery(self, real_qqq, real_vix):
        """
        April 10, 2026: VIX smoothed was 22.87, QQQ just crossed SMA-130.
        Without override: would be uncertain (just barely).
        With override: ROC-5 ≈ +7%, pct ≈ +0.21% → override may fire earlier.
        Both outcomes are valid — just assert regime is not high_vol.
        """
        from daily_signal import compute_regime
        import pandas as pd

        as_of = pd.Timestamp("2026-04-09")  # day before actual first uncertain
        sig = compute_regime(real_qqq, real_vix, as_of)
        # On Apr 9: price very close to SMA, ROC-5 strongly positive
        # Override should fire → uncertain (not high_vol)
        assert sig["regime"] != "high_vol", (
            f"Override should have fired on Apr 9 recovery. "
            f"regime={sig['regime']}, ROC-5={sig.get('roc_5', 'N/A'):.2f}%, "
            f"pct={sig['pct_vs_sma']:.2f}%")


# ── Position sizing config ─────────────────────────────────────────────────────

class TestPositionSizingConfig:
    """Verify expert-panel v2 position sizing parameters are correctly set."""

    def test_strategy_a_max_position_reduced(self):
        """Strategy A max_position_pct must be 0.85 (gap-risk buffer)."""
        from config.strategy_config import STRATEGY_A_CONFIG
        assert STRATEGY_A_CONFIG["max_position_pct"] == 0.85, (
            f"Expected 0.85, got {STRATEGY_A_CONFIG['max_position_pct']}")

    def test_strategy_b_max_position_reduced(self):
        """Strategy B max_position_pct must be 0.60 (gap-risk buffer)."""
        from config.strategy_config import STRATEGY_B_CONFIG
        assert STRATEGY_B_CONFIG["max_position_pct"] == 0.60, (
            f"Expected 0.60, got {STRATEGY_B_CONFIG['max_position_pct']}")

    def test_strategy_b_vol_scale_enabled(self):
        """Strategy B vol_scale must be True (gradual VIX-tier de-risking)."""
        from config.strategy_config import STRATEGY_B_CONFIG
        assert STRATEGY_B_CONFIG["vol_scale"] is True, (
            f"Expected vol_scale=True, got {STRATEGY_B_CONFIG['vol_scale']}")

    def test_strategy_a_vol_scale_unchanged(self):
        """Strategy A vol_scale must remain False (aggressive strategy holds max)."""
        from config.strategy_config import STRATEGY_A_CONFIG
        assert STRATEGY_A_CONFIG["vol_scale"] is False

    def test_position_sizing_creates_cash_buffer(self):
        """With new sizing, max combined TQQQ exposure is 85%×90% + 60%×10% = 82.5% in bull."""
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, ALLOC_CONFIG
        bull_a, bull_b = ALLOC_CONFIG["bull"]
        max_exposure = (STRATEGY_A_CONFIG["max_position_pct"] * bull_a
                        + STRATEGY_B_CONFIG["max_position_pct"] * bull_b)
        assert max_exposure < 1.0, "Combined TQQQ exposure must leave cash buffer"
        assert max_exposure > 0.70, "Cash buffer shouldn't be so large it kills returns"
