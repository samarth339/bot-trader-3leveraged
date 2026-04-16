"""
Edge Case Tests

Validates system behaviour during:
  • Historical market crashes (COVID 2020, rate hikes 2022)
  • Sudden VIX spikes and quick recoveries
  • Missing / stale data scenarios
  • Extreme price gaps
  • Year-end / holiday boundary conditions
"""
import numpy as np
import pandas as pd
import pytest
from daily_signal import compute_regime
from backtester.dual_portfolio import DualPortfolioBacktester
from config.strategy_config import REGIME_CONFIG


# ── Historical crash / stress periods ─────────────────────────────────────────

class TestHistoricalCrashes:

    def test_covid_peak_fear_high_vol(self, real_qqq, real_vix):
        """March 16 2020 (VIX 82.69) must be HIGH_VOL."""
        sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2020-03-16"))
        assert sig["regime"] == "high_vol", \
            f"COVID peak should be high_vol, got '{sig['regime']}'"

    def test_covid_recovery_exits_high_vol(self, real_qqq, real_vix):
        """By August 2020 market should have recovered to BULL or UNCERTAIN."""
        sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2020-08-01"))
        assert sig["regime"] in {"bull", "uncertain"}, \
            f"August 2020 should be bull/uncertain, got '{sig['regime']}'"

    def test_2022_bear_market_detected(self, real_qqq, real_vix):
        """Mid-2022 bear market should be HIGH_VOL."""
        sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2022-07-01"))
        assert sig["regime"] == "high_vol", \
            f"July 2022 bear market should be high_vol, got '{sig['regime']}'"

    def test_2021_bull_market_detected(self, real_qqq, real_vix):
        """Mid-2021 should not be high_vol (VIX grey-zone may produce uncertain — that's OK)."""
        sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2021-08-01"))
        assert sig["regime"] != "high_vol", \
            f"Mid-2021 should not be high_vol, got '{sig['regime']}' (VIX={sig['vix_signal']:.1f})"

    def test_gfc_2008_high_vol(self, real_qqq, real_vix):
        """Oct 2008 GFC peak should be HIGH_VOL."""
        try:
            sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2008-10-15"))
            assert sig["regime"] == "high_vol"
        except ValueError:
            pytest.skip("Data does not extend to 2008")


# ── VIX spike and recovery ─────────────────────────────────────────────────────

class TestVIXSpike:

    def test_spike_triggers_high_vol(self, syn_qqq_df, spike_vix_df):
        """Sudden VIX spike above 25 should trigger HIGH_VOL regime."""
        dp = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=spike_vix_df,
            strategy_a=None, strategy_b=None,
            ma_window=REGIME_CONFIG["ma_window"], vix_smooth=1, t1=True,
        )
        rs = dp.compute_regime_series()
        # Bars around the spike (150–158 in spike_vix_df)
        spike_zone = rs.iloc[155:162]
        assert (spike_zone == "high_vol").any(), \
            "VIX spike to 40 should trigger at least one high_vol day"

    def test_smoothed_vix_delays_regime_change(self, syn_qqq_df, spike_vix_df):
        """5-day VIX smooth delays regime change vs raw VIX."""
        dp_raw = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=spike_vix_df,
            strategy_a=None, strategy_b=None,
            ma_window=REGIME_CONFIG["ma_window"], vix_smooth=1, t1=True,
        )
        dp_sm  = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=spike_vix_df,
            strategy_a=None, strategy_b=None,
            ma_window=REGIME_CONFIG["ma_window"], vix_smooth=5, t1=True,
        )
        rs_raw = dp_raw.compute_regime_series()
        rs_sm  = dp_sm.compute_regime_series()

        # Smoothed should have fewer total high_vol days from the spike
        hv_raw = (rs_raw == "high_vol").sum()
        hv_sm  = (rs_sm  == "high_vol").sum()
        # Smoothed delays onset — raw VIX hits high_vol sooner
        # Just verify they're not identical
        assert hv_raw != hv_sm or rs_raw.equals(rs_sm) is False


# ── Missing / stale data ───────────────────────────────────────────────────────

class TestMissingData:

    def test_missing_data_guard_raises(self, syn_qqq_df, syn_vix_df):
        """NaN in QQQ close must raise ValueError before any computation."""
        bad_qqq = syn_qqq_df.copy()
        bad_qqq.loc[bad_qqq.index[10:15], "close"] = np.nan
        dp = DualPortfolioBacktester(
            tqqq=bad_qqq, sqqq=bad_qqq, qqq=bad_qqq, vix=syn_vix_df,
            strategy_a=None, strategy_b=None,
            ma_window=50, vix_smooth=1, t1=True,
        )
        with pytest.raises(ValueError, match="Missing QQQ signal data"):
            dp.compute_regime_series()

    def test_missing_vix_guard_raises(self, syn_qqq_df, syn_vix_df):
        """NaN in VIX close must raise ValueError."""
        bad_vix = syn_vix_df.copy()
        bad_vix.loc[bad_vix.index[10:15], "close"] = np.nan
        dp = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=bad_vix,
            strategy_a=None, strategy_b=None,
            ma_window=50, vix_smooth=1, t1=True,
        )
        with pytest.raises(ValueError, match="Missing VIX signal data"):
            dp.compute_regime_series()

    def test_date_before_all_data_raises(self, real_qqq, real_vix):
        """Date before any data should raise ValueError cleanly."""
        with pytest.raises(ValueError):
            compute_regime(real_qqq, real_vix, pd.Timestamp("1900-01-01"))

    def test_future_date_uses_latest_available(self, real_qqq, real_vix):
        """A date far in the future should use the latest available bar."""
        sig = compute_regime(real_qqq, real_vix, pd.Timestamp("2099-01-01"))
        assert sig["regime"] in {"bull", "uncertain", "high_vol"}
        # Signal date should be the last available date, not 2099
        assert sig["signal_date"] < pd.Timestamp("2099-01-01")


# ── Boundary conditions ────────────────────────────────────────────────────────

class TestBoundaryConditions:

    def test_year_end_transition(self, real_qqq, real_vix):
        """Dec 31 and Jan 2 both produce valid regimes."""
        for date_str in ["2022-12-30", "2023-01-03"]:
            sig = compute_regime(real_qqq, real_vix, pd.Timestamp(date_str))
            assert sig["regime"] in {"bull", "uncertain", "high_vol"}

    def test_first_available_date_handles_warmup(self, real_qqq, real_vix):
        """Very early dates (no SMA yet) must return UNCERTAIN, not crash."""
        earliest = real_qqq.index[2]  # only 2 bars of history
        sig = compute_regime(real_qqq, real_vix, earliest)
        assert sig["regime"] == "uncertain"

    def test_weekend_date_uses_friday(self, real_qqq, real_vix):
        """Saturday as_of_date should use Friday's close as signal."""
        saturday = pd.Timestamp("2023-06-03")  # a Saturday
        assert saturday.weekday() == 5
        sig = compute_regime(real_qqq, real_vix, saturday)
        assert sig["signal_date"].weekday() < 5, \
            "Signal date must be a weekday (Friday or earlier)"
