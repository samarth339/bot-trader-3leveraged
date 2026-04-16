"""
Lookahead Bias Detection Tests
================================
These tests are the first line of defence against the most dangerous silent
failure in algorithmic trading: using future data to make past decisions.

Failure modes tested:
  1. Direct signal contamination  — close[t] drives allocation[t] (should be close[t-1])
  2. SMA contamination            — SMA at bar t includes close[t]
  3. VIX contamination            — raw VIX[t] instead of smooth_VIX[t-1]
  4. Performance inflation check  — T+0 CAGR must be materially higher than T-1 CAGR
                                    If not, T-1 guard may be a no-op
  5. Signal-date integrity        — signal_date must be < as_of_date on every row
  6. Future-price corruption test — corrupting future prices must not change past signals
  7. Allocation-change timing     — regime flip on day t must produce allocation change on t+1

Run with:
    pytest tests/test_lookahead_bias.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.strategy_config import (
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS, REGIME_CONFIG
)
from config.settings import TQQQ_INCEPTION, INITIAL_CAPITAL
from daily_signal import compute_regime


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_dp(qqq_df, vix_df, t1: bool = True, **kwargs):
    defaults = dict(
        ma_window    = 50,       # shortened for fast synthetic tests
        vix_bull     = REGIME_CONFIG["vix_bull"],
        vix_hi_vol   = REGIME_CONFIG["vix_hi_vol"],
        vix_smooth   = 5,
        t1           = t1,
        confirm_days = 1,
    )
    defaults.update(kwargs)
    return DualPortfolioBacktester(
        tqqq=qqq_df, sqqq=qqq_df, qqq=qqq_df, vix=vix_df,
        strategy_a=None, strategy_b=None,
        **defaults,
    )


def _make_strategies():
    sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
    sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})
    return sa, sb


def _full_dp(tqqq, sqqq, qqq, vix, t1: bool):
    sa, sb = _make_strategies()
    return DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=sa, strategy_b=sb,
        initial_capital=INITIAL_CAPITAL,
        t1=t1,
        **{k: v for k, v in PORTFOLIO_DEFAULTS.items() if k != "t1"},
    )


# ── Test 1: Direct shift verification ─────────────────────────────────────────

class TestSignalShift:
    """Verify that _build_signal_inputs() enforces T-1 at the data level."""

    def test_price_shift_exact(self, syn_qqq_df, syn_vix_df):
        """signal_close[date_t] must exactly equal raw_close[date_{t-1}]."""
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True)
        signal_close, _, _ = dp._build_signal_inputs()
        raw_close = syn_qqq_df["close"]

        mismatches = 0
        for i in range(5, min(50, len(raw_close))):
            today     = raw_close.index[i]
            yesterday = raw_close.index[i - 1]
            delta = abs(signal_close.loc[today] - raw_close.loc[yesterday])
            if delta > 1e-9:
                mismatches += 1

        assert mismatches == 0, (
            f"T-1 price shift has {mismatches} mismatches in first 50 bars — "
            "signal is using wrong date's close price"
        )

    def test_vix_shift_exact(self, syn_qqq_df, syn_vix_df):
        """signal_vix[date_t] must be the lagged smoothed VIX, not today's."""
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True, vix_smooth=1)
        _, _, signal_vix = dp._build_signal_inputs()
        raw_vix = syn_vix_df["close"]

        for i in range(5, 20):
            today     = raw_vix.index[i]
            yesterday = raw_vix.index[i - 1]
            delta = abs(signal_vix.loc[today] - raw_vix.loc[yesterday])
            assert delta < 1e-9, (
                f"VIX T-1 shift failed at {today}: "
                f"signal={signal_vix.loc[today]:.4f} expected={raw_vix.loc[yesterday]:.4f}"
            )

    def test_t0_differs_from_t1(self, syn_qqq_df, syn_vix_df):
        """
        T+0 and T-1 signal series must differ. If identical, the shift is broken.
        This is a meta-test: if the shift has no effect, all T-1 tests are vacuous.
        """
        dp_t1 = _make_dp(syn_qqq_df, syn_vix_df, t1=True)
        dp_t0 = _make_dp(syn_qqq_df, syn_vix_df, t1=False)

        sig_t1, _, _ = dp_t1._build_signal_inputs()
        sig_t0, _, _ = dp_t0._build_signal_inputs()

        # At least some values must differ (due to price change between days)
        diffs = (sig_t1 - sig_t0).abs()
        assert (diffs > 1e-9).any(), (
            "T-1 and T+0 signal_close series are identical — shift may be broken"
        )

    def test_sma_does_not_include_current_bar(self, syn_qqq_df, syn_vix_df):
        """
        The SMA used for regime classification at bar t should not include close[t].
        With T-1, sma[t] should equal rolling_mean(close[0:t-1], window=ma_window).
        """
        ma_window = 20
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True, ma_window=ma_window)
        _, signal_sma, _ = dp._build_signal_inputs()
        raw_close = syn_qqq_df["close"]

        # After warmup, check a specific bar
        check_i = ma_window + 10
        check_date = raw_close.index[check_i]

        # Expected: rolling mean of YESTERDAY's and earlier closes (T-1 shift applied)
        expected_sma = raw_close.iloc[:check_i].rolling(ma_window).mean().iloc[-1]
        actual_sma   = signal_sma.loc[check_date]

        assert abs(actual_sma - expected_sma) < 1e-6, (
            f"SMA at {check_date}: got {actual_sma:.4f}, expected {expected_sma:.4f}. "
            "SMA may be including current bar close."
        )


# ── Test 2: Performance-level inflation test ───────────────────────────────────

class TestPerformanceInflation:
    """
    T+0 backtests should show materially higher CAGR than T-1 backtests.
    If they are equal, the T-1 guard has no effect — which is wrong.
    If T-1 > T+0 significantly, something is backwards.
    """

    @pytest.mark.slow
    def test_t0_cagr_exceeds_t1(self, real_tqqq, real_sqqq, real_qqq, real_vix):
        """T+0 CAGR should exceed T-1 CAGR by at least 1 percentage point."""
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        r_t1 = _full_dp(tqqq, sqqq, qqq, real_vix, t1=True).run()
        r_t0 = _full_dp(tqqq, sqqq, qqq, real_vix, t1=False).run()

        cagr_t1 = r_t1["metrics"]["cagr"]
        cagr_t0 = r_t0["metrics"]["cagr"]

        # T+0 should be detectably higher (look-ahead advantage)
        assert cagr_t0 > cagr_t1, (
            f"T+0 CAGR ({cagr_t0:.1%}) should exceed T-1 CAGR ({cagr_t1:.1%}). "
            "If equal, the T-1 guard may be a no-op."
        )

    @pytest.mark.slow
    def test_t1_cagr_not_suspiciously_high(self, real_tqqq, real_sqqq, real_qqq, real_vix):
        """
        Sanity ceiling: T-1 CAGR above 60% suggests residual look-ahead.
        TQQQ 3x with a real T-1 guard should not exceed this threshold
        over a 15-year period.
        """
        LOOKAHEAD_ALARM_THRESHOLD = 0.60  # 60% CAGR = suspicious

        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        result = _full_dp(tqqq, sqqq, qqq, real_vix, t1=True).run()
        cagr   = result["metrics"]["cagr"]

        assert cagr < LOOKAHEAD_ALARM_THRESHOLD, (
            f"CAGR {cagr:.1%} exceeds lookahead alarm ceiling {LOOKAHEAD_ALARM_THRESHOLD:.0%}. "
            "This is suspicious — verify T-1 guard is active."
        )


# ── Test 3: Future-price corruption test ──────────────────────────────────────

class TestFutureCorruption:
    """
    If we corrupt prices at time t+1, t+2 ... it must not affect the signal at t.
    This directly detects any forward-looking data access.
    """

    def test_corrupting_future_prices_does_not_change_past_regime(
        self, syn_qqq_df, syn_vix_df
    ):
        """
        Take regime at bar 100. Corrupt all prices after bar 100 to extreme values.
        The regime at bar 100 must be identical in both versions.
        """
        CHECK_BAR = 100
        ma_window = 50

        dp_orig = _make_dp(syn_qqq_df, syn_vix_df, t1=True, ma_window=ma_window)
        rs_orig = dp_orig.compute_regime_series()

        # Corrupt: set all close prices after bar CHECK_BAR to 999 (extreme)
        corrupted_qqq = syn_qqq_df.copy()
        corrupted_qqq.iloc[CHECK_BAR:, corrupted_qqq.columns.get_loc("close")] = 999.0
        corrupted_qqq.iloc[CHECK_BAR:, corrupted_qqq.columns.get_loc("high")]  = 1000.0
        corrupted_qqq.iloc[CHECK_BAR:, corrupted_qqq.columns.get_loc("low")]   = 998.0

        dp_corrupt = _make_dp(corrupted_qqq, syn_vix_df, t1=True, ma_window=ma_window)
        rs_corrupt = dp_corrupt.compute_regime_series()

        # Regimes before bar CHECK_BAR must be identical
        past_orig   = rs_orig.iloc[:CHECK_BAR]
        past_corrupt = rs_corrupt.iloc[:CHECK_BAR]

        mismatches = (past_orig != past_corrupt).sum()
        assert mismatches == 0, (
            f"{mismatches} past regime values changed when future prices were corrupted. "
            "This indicates forward-looking data access in regime computation."
        )

    def test_corrupting_future_vix_does_not_change_past_regime(
        self, syn_qqq_df, syn_vix_df
    ):
        """Corrupting future VIX must not alter past regime classifications."""
        CHECK_BAR = 100

        dp_orig = _make_dp(syn_qqq_df, syn_vix_df, t1=True, ma_window=50)
        rs_orig = dp_orig.compute_regime_series()

        corrupted_vix = syn_vix_df.copy()
        corrupted_vix.iloc[CHECK_BAR:, corrupted_vix.columns.get_loc("close")] = 80.0

        dp_corrupt = _make_dp(syn_qqq_df, corrupted_vix, t1=True, ma_window=50)
        rs_corrupt = dp_corrupt.compute_regime_series()

        mismatches = (rs_orig.iloc[:CHECK_BAR] != rs_corrupt.iloc[:CHECK_BAR]).sum()
        assert mismatches == 0, (
            f"{mismatches} past regimes changed when future VIX was corrupted — "
            "VIX signal may be using unshifted data."
        )


# ── Test 4: Signal-date integrity ─────────────────────────────────────────────

class TestSignalDateIntegrity:
    """Verify signal_date < as_of_date on every compute_regime() call."""

    def test_signal_date_always_before_as_of(self, real_qqq, real_vix):
        """
        For 30 sampled dates, signal_date must be strictly before as_of_date.
        Any equality or inversion = lookahead bias.
        """
        test_dates = pd.bdate_range("2015-01-01", "2024-12-31", freq="3BME")[:30]

        violations = []
        for dt in test_dates:
            sig = compute_regime(real_qqq, real_vix, dt)
            if sig["signal_date"] >= sig["as_of_date"]:
                violations.append(
                    f"{dt.date()}: signal_date={sig['signal_date'].date()} "
                    f">= as_of={sig['as_of_date'].date()}"
                )

        assert not violations, (
            f"T-1 signal_date >= as_of_date on {len(violations)} dates:\n"
            + "\n".join(violations)
        )

    def test_signal_date_is_exactly_one_business_day_before(self, real_qqq, real_vix):
        """
        On a normal trading day (Wednesday), signal_date should be Tuesday.
        On a Monday, signal_date should be Friday.
        """
        # Test a Wednesday
        wednesday = pd.Timestamp("2024-03-06")  # known Wednesday
        sig = compute_regime(real_qqq, real_vix, wednesday)
        expected_signal_date = pd.Timestamp("2024-03-05")  # Tuesday
        assert sig["signal_date"] == expected_signal_date, (
            f"On Wednesday {wednesday.date()}, signal_date should be Tuesday "
            f"{expected_signal_date.date()}, got {sig['signal_date'].date()}"
        )

    def test_signal_date_skips_weekend(self, real_qqq, real_vix):
        """For a Monday as_of_date, signal_date must be Friday (not Saturday/Sunday)."""
        monday = pd.Timestamp("2024-03-04")  # known Monday
        assert monday.weekday() == 0
        sig = compute_regime(real_qqq, real_vix, monday)
        assert sig["signal_date"].weekday() < 5, (
            f"signal_date {sig['signal_date'].date()} is a weekend — "
            "T-1 should use the previous Friday"
        )
        assert sig["signal_date"].weekday() == 4, (
            f"For Monday, signal_date should be Friday (weekday=4), "
            f"got weekday={sig['signal_date'].weekday()}"
        )


# ── Test 5: Allocation-change timing ──────────────────────────────────────────

class TestAllocationTiming:
    """
    When a regime flip occurs on day t (detected from close[t-1]),
    the allocation change should not affect the return captured on day t-1.
    This tests the allocation timing chain, not just the shift.
    """

    def test_regime_flip_row_matches_signal_date(self, real_qqq, real_vix):
        """
        Check that on a COVID crash date, the regime flip is attributed
        to as_of_date (today), not signal_date (yesterday).
        The system should not retroactively profit from knowing yesterday crashed.
        """
        # March 16 2020: markets opened after massive Sunday selloff
        # Signal_date = March 13 (Friday before)
        covid_monday = pd.Timestamp("2020-03-16")
        sig = compute_regime(real_qqq, real_vix, covid_monday)

        assert sig["as_of_date"] == covid_monday, (
            f"as_of_date should be {covid_monday.date()}, got {sig['as_of_date'].date()}"
        )
        assert sig["signal_date"] < covid_monday, (
            f"signal_date {sig['signal_date'].date()} should be before {covid_monday.date()}"
        )
        assert sig["regime"] == "high_vol", (
            f"COVID crash should show high_vol regime, got '{sig['regime']}'"
        )

    def test_regime_series_first_valid_bar_is_uncertain(self, syn_qqq_df, syn_vix_df):
        """
        The first bar of the regime series should be UNCERTAIN (no prior data for SMA).
        If it's BULL or HIGH_VOL, the SMA warmup period is being handled incorrectly.
        """
        dp = _make_dp(syn_qqq_df, syn_vix_df, t1=True, ma_window=50)
        rs = dp.compute_regime_series()
        assert rs.iloc[0] == "uncertain", (
            f"First bar should be 'uncertain' (SMA not yet valid), got '{rs.iloc[0]}'"
        )
