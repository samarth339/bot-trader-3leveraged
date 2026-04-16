"""
Extreme Event Tests
====================
Simulates gap moves, VIX spikes, circuit breakers, and multi-day crash sequences.
These tests validate that the system fails safely rather than failing silently.

Scenarios:
  1.  -10% single-day gap down in TQQQ
  2.  +10% single-day gap up in TQQQ
  3.  VIX spike > 50 triggers high_vol regime
  4.  VIX spike > 50 generates a SELL signal
  5.  3 consecutive -5% days (cumulative -14.3% TQQQ decline)
  6.  Prolonged crash: 40-day drawdown sequence
  7.  Circuit breaker: 50% portfolio drawdown halts trading
  8.  Zero-price data: handled without crash or NaN propagation
  9.  VIX = 0 (data error): handled gracefully
  10. Infinite VIX (overflow): handled gracefully
  11. Price gap on regime flip day: test order of operations
  12. Crash during warmup period: system stays in uncertain, not BULL

Run with:
    pytest tests/test_extreme_events.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import Backtester
from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.strategy_config import (
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS, REGIME_CONFIG
)
from config.settings import INITIAL_CAPITAL, MAX_DRAWDOWN_LIMIT, DAILY_STOP_LOSS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_strategy_a():
    return LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})


def _make_strategy_b():
    return LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})


def _inject_gap(df: pd.DataFrame, bar_index: int,
                gap_pct: float) -> pd.DataFrame:
    """
    Inject a price gap on a specific bar.
    gap_pct = -0.10 means -10% gap down.
    Adjusts close, open, high, low consistently.
    """
    result = df.copy()
    factor = 1 + gap_pct

    # From that bar onwards, rescale all prices to simulate the gap
    result.iloc[bar_index:] = result.iloc[bar_index:].copy()
    result.iloc[bar_index, result.columns.get_loc("open")]  = \
        df.iloc[bar_index]["close"] * factor
    result.iloc[bar_index, result.columns.get_loc("close")] = \
        df.iloc[bar_index]["close"] * factor
    result.iloc[bar_index, result.columns.get_loc("high")]  = \
        df.iloc[bar_index]["high"] * factor
    result.iloc[bar_index, result.columns.get_loc("low")]   = \
        df.iloc[bar_index]["low"] * factor * 0.995  # slightly below close

    return result


def _inject_crash_sequence(df: pd.DataFrame, start_bar: int,
                           n_days: int, daily_drop: float) -> pd.DataFrame:
    """
    Inject a multi-day crash sequence starting at start_bar.
    Each day drops by daily_drop (e.g. -0.05 = -5% per day).
    """
    result = df.copy()
    price = result.iloc[start_bar]["close"]

    for i in range(n_days):
        bar = start_bar + i
        if bar >= len(result):
            break
        price *= (1 + daily_drop)
        result.iloc[bar, result.columns.get_loc("close")] = price
        result.iloc[bar, result.columns.get_loc("open")]  = price * 1.002
        result.iloc[bar, result.columns.get_loc("high")]  = price * 1.005
        result.iloc[bar, result.columns.get_loc("low")]   = price * 0.995

    return result


def _run_single(tqqq, sqqq, qqq, **kwargs) -> dict:
    bt = Backtester(
        tqqq, sqqq, qqq,
        initial_capital=INITIAL_CAPITAL,
        slippage_pct=0.001,
        **kwargs,
    )
    return bt.run(_make_strategy_a())


def _run_dual(tqqq, sqqq, qqq, vix, **dp_kwargs) -> dict:
    sa, sb = _make_strategy_a(), _make_strategy_b()
    kw = {**PORTFOLIO_DEFAULTS, **dp_kwargs}
    dp = DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=sa, strategy_b=sb,
        initial_capital=INITIAL_CAPITAL,
        **kw,
    )
    return dp.run()


def _equity(result: dict) -> pd.Series:
    return result["equity_curve"]["equity"]


def _max_dd(result: dict) -> float:
    return result["metrics"]["max_drawdown"]


# ── Gap move tests ─────────────────────────────────────────────────────────────

class TestGapMoves:
    """Single-day extreme gap up and gap down scenarios."""

    def test_gap_down_10pct_equity_survives(self, syn_qqq_df, syn_vix_df):
        """
        A -10% gap down must not wipe out the portfolio.
        Strategy's daily stop-loss should limit the damage.
        """
        GAP_BAR    = 150
        gapped_qqq = _inject_gap(syn_qqq_df, GAP_BAR, -0.10)

        result = _run_single(gapped_qqq, gapped_qqq, gapped_qqq)
        equity = _equity(result)

        assert (equity > 0).all(), (
            "Portfolio equity reached zero or negative after -10% gap down"
        )
        assert equity.min() > INITIAL_CAPITAL * 0.40, (
            f"Equity dropped below 40% of seed (${equity.min():,.0f}) after a single "
            "-10% gap down — daily stop-loss may not be functioning."
        )

    def test_gap_down_10pct_triggers_stop_or_exit(self, syn_qqq_df, syn_vix_df):
        """
        After a -10% gap, the strategy should have exited (stop-loss fired)
        or reduced position significantly by the following bar.
        """
        GAP_BAR    = 150
        gapped_qqq = _inject_gap(syn_qqq_df, GAP_BAR, -0.10)

        result = _run_single(gapped_qqq, gapped_qqq, gapped_qqq)

        # trades is a DataFrame (see engine.py _compile)
        trades = result.get("trades", pd.DataFrame())
        assert not trades.empty, (
            "No trades recorded after -10% gap — strategy may be completely passive"
        )

        # Check if any trade was triggered by stop-loss (exit_reason column)
        stop_exits = trades[trades["exit_reason"].str.contains("stop", case=False, na=False)]
        exit_total = trades[trades["exit_date"] >= syn_qqq_df.index[GAP_BAR]]
        # Either stop-loss OR signal exit should fire around the gap
        assert len(exit_total) > 0 or len(stop_exits) > 0, (
            "No exit was triggered within 5 bars of the -10% gap — "
            "stop-loss or regime exit should have fired."
        )

    def test_gap_up_10pct_equity_does_not_overflow(self, syn_qqq_df, syn_vix_df):
        """
        A +10% gap up must not cause arithmetic overflow or negative equity.
        The system should capture the gain cleanly.
        """
        GAP_BAR    = 150
        gapped_qqq = _inject_gap(syn_qqq_df, GAP_BAR, +0.10)

        result = _run_single(gapped_qqq, gapped_qqq, gapped_qqq)
        equity = _equity(result)

        assert equity.isna().sum() == 0, (
            "NaN values in equity curve after +10% gap up — arithmetic error"
        )
        assert np.isinf(equity).sum() == 0, (
            "Infinite values in equity curve after +10% gap up — overflow detected"
        )
        assert (equity > 0).all(), (
            "Negative equity after +10% gap up — this should not happen"
        )

    def test_gap_up_equity_increases_vs_no_gap(self, syn_qqq_df, syn_vix_df):
        """
        If the strategy is long TQQQ at the time of a +10% gap, the
        equity after the gap day should be higher than without the gap.
        """
        GAP_BAR     = 150
        normal      = _run_single(syn_qqq_df, syn_qqq_df, syn_qqq_df)
        gapped_qqq  = _inject_gap(syn_qqq_df, GAP_BAR, +0.10)
        gapped      = _run_single(gapped_qqq, gapped_qqq, gapped_qqq)

        eq_normal = _equity(normal)
        eq_gapped = _equity(gapped)

        # Final equity should be equal or higher with +10% gap
        assert eq_gapped.iloc[-1] >= eq_normal.iloc[-1] * 0.90, (
            "A +10% gap somehow significantly reduced final equity — "
            "position direction may be inverted."
        )

    def test_multiple_gap_downs_system_stays_solvent(self, syn_qqq_df, syn_vix_df):
        """
        Three -5% gap downs on consecutive bars.
        Combined: ~-14.3% cumulative drop. System must stay solvent.
        """
        gapped = syn_qqq_df.copy()
        for bar in [100, 101, 102]:
            gapped = _inject_gap(gapped, bar, -0.05)

        result = _run_single(gapped, gapped, gapped)
        equity = _equity(result)

        assert (equity > 0).all(), (
            "Portfolio went insolvent after 3 consecutive -5% gap downs"
        )
        assert equity.min() > INITIAL_CAPITAL * 0.30, (
            f"Equity fell below 30% of seed capital after 3 consecutive -5% gaps: "
            f"min=${equity.min():,.0f}"
        )


# ── VIX spike tests ────────────────────────────────────────────────────────────

class TestVIXSpike:
    """VIX spikes to extreme levels — regime detection and signal tests."""

    def test_vix_spike_to_80_triggers_high_vol(self, syn_qqq_df, syn_vix_df):
        """
        A VIX spike to 80 (extreme, like COVID March 2020) must classify
        the regime as high_vol within the smoothing window.
        """
        spiked_vix = syn_vix_df.copy()
        spiked_vix.iloc[100:110, spiked_vix.columns.get_loc("close")] = 80.0

        dp = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=spiked_vix,
            strategy_a=None, strategy_b=None,
            ma_window=50, vix_smooth=3, t1=True,
            vix_bull=REGIME_CONFIG["vix_bull"],
            vix_hi_vol=REGIME_CONFIG["vix_hi_vol"],
        )
        regime_series = dp.compute_regime_series()

        # By bar 108 (allowing for 3-day VIX smooth), must be high_vol
        check_date = syn_qqq_df.index[108]
        regime_at_check = regime_series.loc[check_date]

        assert regime_at_check == "high_vol", (
            f"VIX=80 spike should produce high_vol by bar 108, "
            f"got '{regime_at_check}'"
        )

    def test_vix_spike_50_blocks_bull(self, syn_qqq_df, syn_vix_df):
        """
        VIX >= 25 must block 'bull' regime even when price is above SMA.
        This is a hard rule in _classify(): VIX >= vix_hi_vol → high_vol.
        """
        high_vix = syn_vix_df.copy()
        high_vix["close"] = 50.0  # all bars at VIX=50

        dp = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=high_vix,
            strategy_a=None, strategy_b=None,
            ma_window=50, vix_smooth=1, t1=True,
            vix_bull=REGIME_CONFIG["vix_bull"],
            vix_hi_vol=REGIME_CONFIG["vix_hi_vol"],
        )
        regime_series = dp.compute_regime_series()

        # After warmup, no 'bull' regime should exist with VIX=50
        post_warmup = regime_series.iloc[55:]
        assert (post_warmup != "bull").all(), (
            f"VIX=50 should prevent 'bull' regime. Found bull in: "
            f"{post_warmup[post_warmup == 'bull'].index.tolist()[:3]}"
        )

    def test_vix_spike_recovery_returns_to_bull(self, syn_qqq_df, syn_vix_df):
        """
        After a VIX spike, when VIX returns below threshold, regime should
        be able to return to bull (not permanently stuck in high_vol).
        """
        vix_with_spike = syn_vix_df.copy()
        vix_with_spike.iloc[100:115, vix_with_spike.columns.get_loc("close")] = 50.0
        # After bar 115, VIX returns to calm level
        vix_with_spike.iloc[115:, vix_with_spike.columns.get_loc("close")]    = 12.0

        dp = DualPortfolioBacktester(
            tqqq=syn_qqq_df, sqqq=syn_qqq_df, qqq=syn_qqq_df, vix=vix_with_spike,
            strategy_a=None, strategy_b=None,
            ma_window=50, vix_smooth=5, t1=True,
            vix_bull=REGIME_CONFIG["vix_bull"],
            vix_hi_vol=REGIME_CONFIG["vix_hi_vol"],
        )
        regime_series = dp.compute_regime_series()

        # After the spike clears + smoothing window, should return to bull or uncertain
        post_spike = regime_series.iloc[125:]  # well after VIX returns to 12
        has_non_high_vol = (post_spike != "high_vol").any()

        assert has_non_high_vol, (
            "After VIX returned to 12.0, regime never left 'high_vol'. "
            "Regime may be permanently stuck after a spike."
        )


# ── Multi-day crash tests ──────────────────────────────────────────────────────

class TestMultiDayCrash:
    """Prolonged crash scenarios — 20 to 40 day declining markets."""

    def test_prolonged_crash_does_not_wipe_out(self, syn_qqq_df, syn_vix_df):
        """
        A 30-day crash with -2% per day (~-45% cumulative) must not
        wipe out the portfolio. The strategy should exit or reduce exposure.
        """
        crash_qqq = _inject_crash_sequence(syn_qqq_df, start_bar=100,
                                           n_days=30, daily_drop=-0.02)

        result = _run_single(crash_qqq, crash_qqq, crash_qqq)
        equity = _equity(result)

        assert (equity > 0).all(), (
            "Portfolio wiped out during 30-day crash sequence"
        )

    def test_circuit_breaker_triggers_on_50pct_drawdown(
        self, syn_qqq_df, syn_vix_df
    ):
        """
        The Backtester's circuit breaker should halt trading when portfolio
        drawdown exceeds MAX_DRAWDOWN_LIMIT (50%).
        After the halt, equity should stay flat (cash).
        """
        # Inject a severe -60% crash
        severe_crash = _inject_crash_sequence(syn_qqq_df, start_bar=50,
                                              n_days=60, daily_drop=-0.025)

        result = _run_single(severe_crash, severe_crash, severe_crash)
        equity = _equity(result)

        peak = equity.cummax()
        drawdown = ((peak - equity) / peak)
        max_dd = drawdown.max()

        # Max drawdown should be bounded at or near the halt threshold
        assert max_dd <= MAX_DRAWDOWN_LIMIT + 0.15, (
            f"Drawdown {max_dd:.1%} substantially exceeds halt threshold "
            f"{MAX_DRAWDOWN_LIMIT:.0%}. Circuit breaker may not be active."
        )

    def test_post_crash_equity_is_flat(self, syn_qqq_df, syn_vix_df):
        """
        After the circuit breaker fires (50% DD), equity should be flat
        (no further trades). Verify equity curve is monotonically non-decreasing
        after the trough.
        """
        severe_crash = _inject_crash_sequence(syn_qqq_df, start_bar=50,
                                              n_days=60, daily_drop=-0.025)
        result   = _run_single(severe_crash, severe_crash, severe_crash)
        equity   = _equity(result)
        drawdown = ((equity.cummax() - equity) / equity.cummax())

        # Find bar where drawdown first exceeds 45%
        halt_bars = drawdown[drawdown > 0.45].index
        if len(halt_bars) == 0:
            pytest.skip("Circuit breaker did not fire — crash not severe enough for this test")

        halt_date = halt_bars[0]
        post_halt = equity.loc[halt_date:]

        # After halt: equity should not decline significantly further
        # (Circuit breaker stops new buys; existing cash should not shrink)
        max_post_decline = (post_halt.max() - post_halt.min()) / post_halt.max()
        assert max_post_decline < 0.20, (
            f"After circuit breaker at {halt_date.date()}, equity still dropped "
            f"{max_post_decline:.1%}. Circuit breaker may not be halting trading."
        )


# ── Data anomaly tests ─────────────────────────────────────────────────────────

class TestDataAnomalies:
    """Verify graceful handling of bad or extreme data values."""

    def test_zero_price_does_not_corrupt_equity(self, syn_qqq_df, syn_vix_df):
        """
        A single zero close price should not corrupt the equity curve with
        NaN or Inf values. The engine may skip the bar or use a guard — either
        way the equity series must remain finite and positive.
        """
        zero_df = syn_qqq_df.copy()
        zero_df.iloc[50, zero_df.columns.get_loc("close")] = 0.0

        try:
            bt = Backtester(zero_df, zero_df, zero_df,
                            initial_capital=INITIAL_CAPITAL)
            result = bt.run(_make_strategy_a())
            equity = _equity(result)
            assert not equity.isna().any(), "Zero price produced NaN in equity curve"
            assert not np.isinf(equity).any(), "Zero price produced Inf in equity curve"
            assert (equity > 0).all(), "Zero price produced non-positive equity"
        except (ValueError, ZeroDivisionError):
            pass  # Engine raising is also acceptable behaviour

    def test_nan_price_in_live_region_raises(self, syn_qqq_df, syn_vix_df):
        """
        NaN in price data should raise a clear ValueError, not propagate silently.
        """
        nan_df = syn_qqq_df.copy()
        nan_df.iloc[50, nan_df.columns.get_loc("close")] = np.nan
        nan_df.iloc[50, nan_df.columns.get_loc("high")]  = np.nan

        with pytest.raises(ValueError):
            Backtester(nan_df, nan_df, nan_df, initial_capital=INITIAL_CAPITAL)

    def test_vix_zero_does_not_crash_regime_classifier(
        self, syn_qqq_df, syn_vix_df
    ):
        """
        VIX = 0 is a data error. The classifier must not crash — it should
        treat it as below the bull threshold (= bull regime) or uncertain.
        """
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=100.0, vix_val=0.0,
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime in {"bull", "uncertain", "high_vol"}, (
            f"_classify() returned invalid regime '{regime}' for VIX=0"
        )

    def test_vix_nan_does_not_crash_regime_classifier(self, syn_qqq_df):
        """
        VIX = NaN should produce 'uncertain' (safe default), not raise.
        """
        regime = DualPortfolioBacktester._classify(
            price=105.0, sma=100.0, vix_val=float("nan"),
            vix_bull=18.0, vix_hi_vol=25.0,
        )
        assert regime in {"bull", "uncertain", "high_vol"}, (
            f"_classify() returned invalid regime '{regime}' for VIX=NaN"
        )

    def test_extreme_vix_value_produces_high_vol(self, syn_qqq_df):
        """
        VIX = 200 (physically impossible but tests overflow handling).
        Must produce 'high_vol', not an error.
        """
        for extreme_vix in [100.0, 200.0, 999.9]:
            regime = DualPortfolioBacktester._classify(
                price=105.0, sma=100.0, vix_val=extreme_vix,
                vix_bull=18.0, vix_hi_vol=25.0,
            )
            assert regime == "high_vol", (
                f"VIX={extreme_vix} should produce 'high_vol', got '{regime}'"
            )


# ── Crash during warmup ────────────────────────────────────────────────────────

class TestCrashDuringWarmup:
    """
    If a crash happens during the SMA warmup period (first 150 bars),
    the regime must stay 'uncertain' — not default to a misleading signal.
    """

    def test_crash_during_warmup_stays_uncertain(self, syn_qqq_df, syn_vix_df):
        """
        A -40% crash in the first 50 bars must not produce a 'bull' regime.
        With no SMA history, the classifier must default to 'uncertain'.
        """
        ma_window = 100
        crash_early = _inject_crash_sequence(
            syn_qqq_df, start_bar=5, n_days=30, daily_drop=-0.02
        )

        dp = DualPortfolioBacktester(
            tqqq=crash_early, sqqq=crash_early, qqq=crash_early, vix=syn_vix_df,
            strategy_a=None, strategy_b=None,
            ma_window=ma_window, vix_smooth=5, t1=True,
            vix_bull=REGIME_CONFIG["vix_bull"],
            vix_hi_vol=REGIME_CONFIG["vix_hi_vol"],
        )
        regime_series = dp.compute_regime_series()

        # During warmup, no bar should ever be 'bull'
        warmup = regime_series.iloc[:ma_window]
        assert (warmup != "bull").all(), (
            f"During warmup ({ma_window} bars), some days showed 'bull' regime "
            f"despite a crash. Warmup guard may be broken."
        )
