"""
Unit Tests — Regime Classifier

Tests the core regime logic in DualPortfolioBacktester:
  • Correct bull / uncertain / high_vol labels
  • T-1 hard guard (no look-ahead)
  • Deterministic assert (no undefined states)
  • VIX smoothing
  • Confirm-days inertia
"""
import numpy as np
import pandas as pd
import pytest
from backtester.dual_portfolio import DualPortfolioBacktester
from config.strategy_config import REGIME_CONFIG


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dp(qqq_df, vix_df, **kwargs):
    """Minimal DualPortfolioBacktester for regime-only tests (no strategies)."""
    defaults = dict(
        ma_window    = REGIME_CONFIG["ma_window"],
        vix_bull     = REGIME_CONFIG["vix_bull"],
        vix_hi_vol   = REGIME_CONFIG["vix_hi_vol"],
        vix_smooth   = 1,
        t1           = True,
        confirm_days = 1,
    )
    defaults.update(kwargs)
    # strategy_a/b are not used — pass None (regime tests only)
    return DualPortfolioBacktester(
        tqqq=qqq_df, sqqq=qqq_df, qqq=qqq_df, vix=vix_df,
        strategy_a=None, strategy_b=None,
        **defaults,
    )


# ── classify() pure-function tests ───────────────────────────────────────────

class TestClassifyPureFunction:

    def test_bull_clear(self):
        """Price well above SMA, VIX calm → BULL."""
        regime = DualPortfolioBacktester._classify(
            price=110.0, sma=100.0, vix_val=14.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "bull"

    def test_high_vol_vix_spike(self):
        """VIX ≥ vix_hi_vol → HIGH_VOL regardless of price vs SMA."""
        regime = DualPortfolioBacktester._classify(
            price=110.0, sma=100.0, vix_val=30.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "high_vol"

    def test_high_vol_price_below_sma(self):
        """Price below SMA → HIGH_VOL even if VIX is moderate."""
        regime = DualPortfolioBacktester._classify(
            price=90.0, sma=100.0, vix_val=20.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "high_vol"

    def test_uncertain_grey_zone(self):
        """Price above SMA but VIX in grey zone (18–25) → UNCERTAIN."""
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=100.0, vix_val=21.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "uncertain"

    def test_uncertain_exact_vix_bull_boundary(self):
        """VIX exactly at vix_bull threshold — not strictly below → UNCERTAIN."""
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=100.0, vix_val=18.0,  # exactly at boundary
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "uncertain"

    def test_high_vol_exact_vix_boundary(self):
        """VIX exactly at vix_hi_vol → HIGH_VOL (inclusive threshold)."""
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=100.0, vix_val=25.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "high_vol"

    def test_nan_sma_returns_uncertain(self):
        """NaN SMA (insufficient history) → UNCERTAIN, never crashes."""
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=float("nan"), vix_val=14.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "uncertain"

    def test_nan_price_returns_uncertain(self):
        """NaN price → UNCERTAIN."""
        regime = DualPortfolioBacktester._classify(
            price=float("nan"), sma=100.0, vix_val=14.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime == "uncertain"

    @pytest.mark.parametrize("price,sma,vix", [
        (110, 100, 14),   # bull
        (90,  100, 20),   # high_vol (below MA)
        (110, 100, 30),   # high_vol (vix spike)
        (105, 100, 21),   # uncertain (grey zone)
        (float("nan"), 100, 14),  # uncertain (no price)
    ])
    def test_all_combos_return_valid_regime(self, price, sma, vix):
        """Exhaustive check: every combination produces a valid regime string."""
        regime = DualPortfolioBacktester._classify(
            price=price, sma=sma, vix_val=vix,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime in {"bull", "uncertain", "high_vol"}


# ── T-1 hard guard ────────────────────────────────────────────────────────────

class TestT1HardGuard:

    def test_signal_inputs_are_shifted(self, syn_qqq_df, syn_vix_df):
        """
        _build_signal_inputs() with t1=True must return series where
        signal_close[date_i] == raw_close[date_{i-1}].
        """
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True)
        signal_close, _, _ = dp._build_signal_inputs()
        raw_close = syn_qqq_df["close"]

        # Compare day 5 onwards (first bar is NaN after shift)
        for i in range(5, 20):
            date_today = raw_close.index[i]
            date_yesterday = raw_close.index[i - 1]
            assert abs(signal_close.loc[date_today] - raw_close.loc[date_yesterday]) < 1e-6, \
                f"T-1 guard failed: signal_close[{date_today}] should equal raw_close[{date_yesterday}]"

    def test_t0_signal_is_not_shifted(self, syn_qqq_df, syn_vix_df):
        """With t1=False, signal_close[date_i] == raw_close[date_i]."""
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=False)
        signal_close, _, _ = dp._build_signal_inputs()
        raw_close = syn_qqq_df["close"]
        for i in range(5, 20):
            date = raw_close.index[i]
            assert abs(signal_close.loc[date] - raw_close.loc[date]) < 1e-6

    def test_t1_regime_differs_from_t0(self, syn_qqq_df, syn_vix_df):
        """
        T-1 and T+0 regime series must differ on at least some days
        (otherwise T-1 guard is a no-op).
        """
        dp_t1 = _make_dp(syn_qqq_df, syn_vix_df, t1=True)
        dp_t0 = _make_dp(syn_qqq_df, syn_vix_df, t1=False)
        s1 = dp_t1.compute_regime_series()
        s0 = dp_t0.compute_regime_series()
        # At minimum 1 bar must differ due to shift
        assert (s1 != s0).any(), "T-1 and T+0 regimes are identical — guard may be broken"

    def test_vix_also_shifted(self, syn_qqq_df, syn_vix_df):
        """signal_vix with t1=True must be 1-bar lagged raw VIX."""
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True, vix_smooth=1)
        _, _, signal_vix = dp._build_signal_inputs()
        raw_vix = syn_vix_df["close"]
        for i in range(5, 20):
            d_today = raw_vix.index[i]
            d_prev  = raw_vix.index[i - 1]
            assert abs(signal_vix.loc[d_today] - raw_vix.loc[d_prev]) < 1e-6


# ── VIX smoothing ─────────────────────────────────────────────────────────────

class TestVIXSmoothing:

    def test_smoothing_reduces_variance(self, syn_qqq_df, spike_vix_df):
        """5-day smoothed VIX has lower std than raw VIX."""
        dp_raw    = _make_dp(syn_qqq_df, spike_vix_df, vix_smooth=1)
        dp_smooth = _make_dp(syn_qqq_df, spike_vix_df, vix_smooth=5)
        _, _, sig_raw    = dp_raw._build_signal_inputs()
        _, _, sig_smooth = dp_smooth._build_signal_inputs()
        assert sig_smooth.std() < sig_raw.std(), \
            "Smoothed VIX should have lower std than raw VIX"

    def test_smooth_window_1_equals_raw(self, syn_qqq_df, syn_vix_df):
        """vix_smooth=1 must produce the same result as no smoothing."""
        dp = _make_dp(syn_qqq_df, syn_vix_df, vix_smooth=1)
        _, _, sig1 = dp._build_signal_inputs()
        # vix_smooth=1 rolling mean is identical to raw (bar-by-bar)
        raw_vix = syn_vix_df["close"].shift(1).ffill()  # T-1 raw
        diff = (sig1 - raw_vix).abs().max()
        assert diff < 1e-6


# ── Regime series & confirm-days ──────────────────────────────────────────────

class TestRegimeSeries:

    def test_early_bars_are_uncertain(self, syn_qqq_df, syn_vix_df):
        """First MA_WINDOW bars must be UNCERTAIN (not enough history for SMA)."""
        ma_window = REGIME_CONFIG["ma_window"]
        dp = _make_dp(syn_qqq_df, syn_vix_df)
        rs = dp.compute_regime_series()
        # First ma_window bars (after T-1 shift) should all be uncertain
        assert (rs.iloc[:ma_window] == "uncertain").all(), \
            f"First {ma_window} bars should all be 'uncertain' — SMA not yet valid"

    def test_regime_series_all_valid(self, syn_qqq_df, syn_vix_df):
        """Every value in regime series must be a valid label."""
        dp = _make_dp(syn_qqq_df, syn_vix_df)
        rs = dp.compute_regime_series()
        invalid = set(rs.unique()) - {"bull", "uncertain", "high_vol"}
        assert not invalid, f"Invalid regime labels found: {invalid}"

    def test_confirm_days_reduces_flips(self, syn_qqq_df, spike_vix_df):
        """With confirm_days=3, regime flips fewer times than confirm_days=1."""
        dp1 = _make_dp(syn_qqq_df, spike_vix_df, confirm_days=1)
        dp3 = _make_dp(syn_qqq_df, spike_vix_df, confirm_days=3)
        rs1 = dp1.compute_regime_series()
        rs3 = dp3.compute_regime_series()
        flips1 = (rs1 != rs1.shift()).sum()
        flips3 = (rs3 != rs3.shift()).sum()
        assert flips3 <= flips1, \
            f"confirm_days=3 should have ≤ flips vs confirm_days=1: {flips3} vs {flips1}"

    def test_high_vix_produces_high_vol_regime(self, bull_qqq_df, high_vix_df):
        """Even a bull-trending price series → HIGH_VOL when VIX is elevated."""
        ma = REGIME_CONFIG["ma_window"]
        dp = _make_dp(bull_qqq_df, high_vix_df)
        rs = dp.compute_regime_series()
        # Skip warmup bars
        late_regimes = rs.iloc[ma + 5:]
        assert (late_regimes == "high_vol").any(), \
            "High VIX should trigger high_vol regime even when price above SMA"
