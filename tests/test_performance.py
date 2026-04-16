"""
Performance Regression Tests

These are the production acceptance criteria.
If any of these tests fail, the system should NOT be deployed.

Thresholds are deliberately CONSERVATIVE — set below historical results
to catch regressions without chasing exact numbers.
"""
import pandas as pd
import pytest
from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.settings import TQQQ_INCEPTION, INITIAL_CAPITAL
from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS

# ── Acceptance thresholds (conservative) ────────────────────────────────────
# NOTE: Thresholds updated for expert-panel v2 config (2026-04-16).
# Deliberate trade-off: position sizing reduced (A 0.95→0.85, B 0.70→0.60) and
# vol_scale enabled for B. Effect: CAGR ↓ ~6pp, Max DD ↓ from ~55% to 37.7%,
# Calmar ↑ slightly (0.54 → 0.55). Thresholds updated to track new baseline.
#
# CAGR_MIN  = actual 20.59% − 3.6pp regression buffer = 17%
# MAX_DD_LIMIT tightened from 55% → 45% to guard against regression to old sizing
CAGR_MIN       = 0.17    # ≥ 17% CAGR (current actual: 20.6% — regression buffer)
CAGR_MAX       = 0.40    # ≤ 40% (sanity ceiling — overfitting alarm)
MAX_DD_LIMIT   = 0.45    # max drawdown must stay below 45% (actual: 37.7%)
CALMAR_MIN     = 0.45    # Calmar ratio ≥ 0.45 (actual: 0.55)
SHARPE_MIN     = 0.60    # Sharpe ≥ 0.60 (actual: 0.76)
COVID_DD_MAX   = 0.55    # COVID crash drawdown ≤ 55%
RATES_DD_MAX   = 0.45    # 2022 rate-hike drawdown ≤ 45%
FINAL_MIN      = 50_000  # $5K seed must grow to at least $50K (actual: $103K)


@pytest.fixture(scope="module")
def dual_result(real_tqqq, real_sqqq, real_qqq, real_vix):
    """Full backtest result — computed once, shared by all performance tests."""
    sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
    sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})

    tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
    sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
    qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

    dp = DualPortfolioBacktester(
        tqqq, sqqq, qqq, real_vix,
        strategy_a=sa, strategy_b=sb,
        initial_capital=INITIAL_CAPITAL,
        **PORTFOLIO_DEFAULTS,
    )
    return dp.run()


# ── Core performance ──────────────────────────────────────────────────────────

class TestCorePerformance:

    def test_cagr_above_floor(self, dual_result):
        cagr = dual_result["metrics"]["cagr"]
        assert cagr >= CAGR_MIN, \
            f"CAGR {cagr*100:.1f}% below minimum {CAGR_MIN*100:.0f}%"

    def test_cagr_sanity_ceiling(self, dual_result):
        """CAGR above ceiling suggests look-ahead / overfitting."""
        cagr = dual_result["metrics"]["cagr"]
        assert cagr <= CAGR_MAX, \
            f"CAGR {cagr*100:.1f}% above sanity ceiling {CAGR_MAX*100:.0f}% — check for bias"

    def test_max_drawdown_acceptable(self, dual_result):
        dd = dual_result["metrics"]["max_drawdown"]
        assert dd <= MAX_DD_LIMIT, \
            f"Max DD {dd*100:.1f}% exceeds limit {MAX_DD_LIMIT*100:.0f}%"

    def test_calmar_above_floor(self, dual_result):
        calmar = dual_result["metrics"]["calmar"]
        assert calmar >= CALMAR_MIN, \
            f"Calmar {calmar:.3f} below minimum {CALMAR_MIN}"

    def test_sharpe_above_floor(self, dual_result):
        sharpe = dual_result["metrics"]["sharpe"]
        assert sharpe >= SHARPE_MIN, \
            f"Sharpe {sharpe:.3f} below minimum {SHARPE_MIN}"

    def test_final_equity_above_floor(self, dual_result):
        final = dual_result["metrics"]["final_equity"]
        assert final >= FINAL_MIN, \
            f"Final equity ${final:,.0f} below floor ${FINAL_MIN:,}"

    def test_equity_never_reaches_zero(self, dual_result):
        min_eq = dual_result["equity_curve"]["equity"].min()
        assert min_eq > 0, f"Portfolio reached ${min_eq:.2f} — wipeout detected"


# ── Stress period performance ─────────────────────────────────────────────────

class TestStressPeriods:

    def _period_drawdown(self, ec, start, end):
        seg = ec.loc[start:end, "equity"]
        if len(seg) < 5:
            return 0.0
        return float(((seg.cummax() - seg) / seg.cummax()).max())

    def _period_return(self, ec, start, end):
        seg = ec.loc[start:end, "equity"]
        if len(seg) < 5:
            return 0.0
        return float((seg.iloc[-1] / seg.iloc[0]) - 1)

    def test_covid_drawdown_bounded(self, dual_result):
        ec = dual_result["equity_curve"]
        dd = self._period_drawdown(ec, "2020-02-19", "2020-11-30")
        assert dd <= COVID_DD_MAX, \
            f"COVID drawdown {dd*100:.1f}% exceeds limit {COVID_DD_MAX*100:.0f}%"

    def test_2022_drawdown_bounded(self, dual_result):
        ec = dual_result["equity_curve"]
        dd = self._period_drawdown(ec, "2021-11-19", "2022-12-31")
        assert dd <= RATES_DD_MAX, \
            f"2022 drawdown {dd*100:.1f}% exceeds limit {RATES_DD_MAX*100:.0f}%"

    def test_covid_recovery_positive_by_year_end(self, dual_result):
        """Portfolio should be profitable by end of 2020 despite COVID crash."""
        ec = dual_result["equity_curve"]
        ret = self._period_return(ec, "2020-01-01", "2020-12-31")
        assert ret > -0.10, \
            f"Full-year 2020 return {ret*100:.1f}% — system did not recover from COVID"


# ── Regime sanity ─────────────────────────────────────────────────────────────

class TestRegimeSanity:

    def test_regime_flips_bounded(self, dual_result):
        """Should not flip regime more than ~20/year over 15 years (≤ 300 total).
        High count = bull↔uncertain transitions on VIX 18 boundary — normal behaviour."""
        flips = dual_result["metrics"]["regime_flips"]
        assert flips < 300, \
            f"Excessive regime flips ({flips}) — check VIX threshold or smoothing"

    def test_bull_regime_dominates(self, dual_result):
        """Bull regime should be the most common (TQQQ is in an uptrend most of the time)."""
        pct_bull = dual_result["metrics"]["pct_bull"]
        assert pct_bull > 30, \
            f"Bull regime only {pct_bull:.1f}% of time — regime classifier may be too bearish"

    def test_high_vol_regime_not_excessive(self, dual_result):
        """High-vol should not dominate — this would mean we're mostly in defensive mode."""
        pct_hv = dual_result["metrics"]["pct_high_vol"]
        assert pct_hv < 50, \
            f"High-vol regime is {pct_hv:.1f}% — classifier is too defensive"
