"""
Regime Noise & Misclassification Tests
=========================================
Simulates regime classification errors (10–20% random misclassification)
and verifies the strategy degrades gracefully rather than catastrophically.

This is a critical robustness test: real-world regimes are never perfectly
classified. A strategy that collapses under 15% noise has fragile regime logic.

Test scenarios:
  1. 10% random regime misclassification — CAGR must not drop >30%
  2. 20% random regime misclassification — CAGR must stay positive
  3. Systematic bias to 'uncertain' — performance floor test
  4. Systematic bias to 'high_vol' — defensive over-reaction test
  5. Flip-back noise (regime bounces one bar then reverts)
  6. VIX threshold sensitivity (±2 point shift)
  7. MA window sensitivity (±10 bars)

Run with:
    pytest tests/test_regime_noise.py -v
"""

import numpy as np
import pandas as pd
import pytest
from typing import Callable

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.strategy_config import (
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS, REGIME_CONFIG
)
from config.settings import TQQQ_INCEPTION, INITIAL_CAPITAL

VALID_REGIMES = ["bull", "uncertain", "high_vol"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_strategies():
    sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
    sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})
    return sa, sb


def _run_dual(tqqq, sqqq, qqq, vix, regime_series=None, **dp_kwargs) -> dict:
    sa, sb = _make_strategies()
    kw = {**PORTFOLIO_DEFAULTS, **dp_kwargs}
    dp = DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=sa, strategy_b=sb,
        initial_capital=INITIAL_CAPITAL,
        **kw,
    )
    return dp.run(regime_series=regime_series)


def _inject_noise(regime_series: pd.Series, noise_rate: float,
                  seed: int = 42) -> pd.Series:
    """
    Randomly misclassify `noise_rate` fraction of bars.
    On a misclassified bar, replace the regime with a randomly chosen
    different regime from VALID_REGIMES.
    """
    rng     = np.random.default_rng(seed)
    noisy   = regime_series.copy()
    n       = len(noisy)
    n_flip  = int(n * noise_rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)

    for i in flip_idx:
        current = noisy.iloc[i]
        choices = [r for r in VALID_REGIMES if r != current]
        noisy.iloc[i] = rng.choice(choices)

    return noisy


def _inject_flipback_noise(regime_series: pd.Series, every_n: int = 10,
                           seed: int = 42) -> pd.Series:
    """
    Every `every_n` bars, flip to a random different regime for 1 bar,
    then revert. Simulates transient misclassifications.
    """
    rng   = np.random.default_rng(seed)
    noisy = regime_series.copy()

    for i in range(every_n, len(noisy) - 1, every_n):
        original = noisy.iloc[i]
        choices  = [r for r in VALID_REGIMES if r != original]
        noisy.iloc[i] = rng.choice(choices)
        # Revert next bar (flip-back)
        noisy.iloc[i + 1] = original

    return noisy


def _cagr(result: dict) -> float:
    return result["metrics"]["cagr"]


def _max_dd(result: dict) -> float:
    return result["metrics"]["max_drawdown"]


# ── Regime noise tests ─────────────────────────────────────────────────────────

class TestRandomRegimeNoise:
    """Random misclassification at 10% and 20% noise rates."""

    @pytest.mark.slow
    def test_10pct_noise_cagr_does_not_collapse(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        At 10% regime misclassification, CAGR should not drop more than 30%
        relative to the noise-free baseline.
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        # Baseline: no noise
        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )
        clean_regime = dp.compute_regime_series()
        r_clean      = dp.run(regime_series=clean_regime)

        # Noisy: 10% misclassified
        noisy_regime = _inject_noise(clean_regime, noise_rate=0.10, seed=42)
        r_noisy      = dp.run(regime_series=noisy_regime)

        cagr_clean = _cagr(r_clean)
        cagr_noisy = _cagr(r_noisy)
        pct_drop   = (cagr_clean - cagr_noisy) / cagr_clean if cagr_clean > 0 else 0

        assert cagr_noisy > 0, (
            f"10% noise drove CAGR to {cagr_noisy:.1%} (negative) — "
            "strategy is catastrophically fragile to regime errors."
        )
        assert pct_drop < 0.30, (
            f"10% noise caused {pct_drop:.0%} CAGR drop "
            f"({cagr_clean:.1%} → {cagr_noisy:.1%}). "
            "Threshold: ≤ 30% relative drop."
        )

    @pytest.mark.slow
    def test_20pct_noise_stays_positive(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        At 20% noise, CAGR must stay positive. A strategy that fails at 20%
        misclassification is too fragile for live deployment.
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )
        regime    = dp.compute_regime_series()
        noisy     = _inject_noise(regime, noise_rate=0.20, seed=77)
        r_noisy   = dp.run(regime_series=noisy)
        cagr      = _cagr(r_noisy)

        assert cagr > 0, (
            f"20% regime noise drove CAGR to {cagr:.1%} — strategy is not robust."
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("noise_rate,min_cagr", [
        (0.05, 0.15),   # 5% noise  → at least 15% CAGR
        (0.10, 0.10),   # 10% noise → at least 10% CAGR
        (0.20, 0.00),   # 20% noise → must stay positive
    ])
    def test_noise_cagr_floor_parametrize(
        self, noise_rate, min_cagr, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """Parametric noise floor: verify each noise level meets its CAGR floor."""
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )
        regime = dp.compute_regime_series()
        noisy  = _inject_noise(regime, noise_rate=noise_rate, seed=42)
        result = dp.run(regime_series=noisy)
        cagr   = _cagr(result)

        assert cagr >= min_cagr, (
            f"At {noise_rate:.0%} noise: CAGR {cagr:.1%} < floor {min_cagr:.1%}"
        )


# ── Systematic regime bias tests ───────────────────────────────────────────────

class TestSystematicRegimeBias:
    """Force all regimes to a single value and measure performance floor."""

    @pytest.mark.slow
    def test_all_uncertain_stays_viable(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        If regime classifier always returns 'uncertain', allocation is 50/50.
        Strategy should still grow capital (not ideal, but not catastrophic).
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )

        regime_always_uncertain = dp.compute_regime_series().copy()
        regime_always_uncertain[:] = "uncertain"
        result = dp.run(regime_series=regime_always_uncertain)
        cagr   = _cagr(result)

        assert cagr > 0.05, (
            f"With constant 'uncertain' regime, CAGR = {cagr:.1%} < 5%. "
            "Strategy B (defensive, 50% weight) should still produce positive returns."
        )

    @pytest.mark.slow
    def test_all_high_vol_stays_solvent(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        If regime always = 'high_vol', allocation is 30% A / 70% B (very defensive).
        Strategy must not lose capital overall (may have low CAGR, that's OK).
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )

        regime_hi_vol = dp.compute_regime_series().copy()
        regime_hi_vol[:] = "high_vol"
        result    = dp.run(regime_series=regime_hi_vol)
        final_eq  = result["metrics"]["final_equity"]

        assert final_eq > INITIAL_CAPITAL, (
            f"Always-high_vol allocation: final equity ${final_eq:,.0f} < "
            f"seed ${INITIAL_CAPITAL:,}. Strategy B alone should grow capital over 15 years."
        )

    @pytest.mark.slow
    def test_all_bull_reasonable_performance(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        With regime always = 'bull', allocation is 75% A / 25% B.
        This is the most aggressive case — CAGR should be high but drawdown too.
        Verify it doesn't produce an unrealistic number (look-ahead sanity check).
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )

        regime_bull = dp.compute_regime_series().copy()
        regime_bull[:] = "bull"
        result = dp.run(regime_series=regime_bull)
        cagr   = _cagr(result)

        # Should be reasonable: real-world TQQQ with 75% exposure
        assert 0 < cagr < 0.80, (
            f"Always-bull CAGR = {cagr:.1%} is outside 0–80% band. "
            "Either strategy is broken or look-ahead is present."
        )


# ── Flipback noise ─────────────────────────────────────────────────────────────

class TestFlipbackNoise:
    """
    Regime bounces 1 bar every N bars and reverts. Tests the confirm_days
    mechanism and overall stability under transient misclassifications.
    """

    @pytest.mark.slow
    def test_flipback_noise_cagr_stable(
        self, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """
        Flipback noise every 10 bars should cause < 20% CAGR degradation.
        This represents a regime classifier that has occasional 1-bar errors.
        """
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )

        clean_regime = dp.compute_regime_series()
        r_clean      = dp.run(regime_series=clean_regime)
        cagr_clean   = _cagr(r_clean)

        noisy_regime = _inject_flipback_noise(clean_regime, every_n=10)
        r_noisy      = dp.run(regime_series=noisy_regime)
        cagr_noisy   = _cagr(r_noisy)

        pct_drop = (cagr_clean - cagr_noisy) / max(cagr_clean, 0.001)
        assert pct_drop < 0.20, (
            f"Flipback noise caused {pct_drop:.0%} CAGR drop "
            f"({cagr_clean:.1%} → {cagr_noisy:.1%}). "
            "Strategy is too sensitive to transient regime misclassification."
        )


# ── VIX threshold sensitivity ──────────────────────────────────────────────────

class TestVIXThresholdSensitivity:
    """
    Perturb the VIX bull and high-vol thresholds by ±2 points.
    Performance should degrade gracefully, not collapse.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("vix_bull_delta,vix_hv_delta", [
        (-2, -2),    # tighter: go to bull less often
        (+2, +2),    # looser: stay in bull more often
        (-2, +2),    # asymmetric: narrow uncertain band
        (+2, -2),    # asymmetric: wide uncertain band
    ])
    def test_vix_threshold_perturbation(
        self, vix_bull_delta, vix_hv_delta,
        real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """±2 point shift in VIX thresholds must keep CAGR above 10%."""
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        base_vix_bull  = REGIME_CONFIG["vix_bull"]    # 18.0
        base_vix_hi    = REGIME_CONFIG["vix_hi_vol"]  # 25.0
        new_vix_bull   = base_vix_bull + vix_bull_delta
        new_vix_hi     = base_vix_hi   + vix_hv_delta

        # Ensure thresholds are ordered correctly
        if new_vix_bull >= new_vix_hi:
            pytest.skip("Invalid threshold configuration (bull >= hi_vol)")

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            vix_bull  = new_vix_bull,
            vix_hi_vol= new_vix_hi,
            **{k: v for k, v in PORTFOLIO_DEFAULTS.items()
               if k not in ("vix_bull", "vix_hi_vol")},
        )
        result = dp.run()
        cagr   = _cagr(result)

        assert cagr > 0.10, (
            f"VIX thresholds shifted by ({vix_bull_delta:+d}, {vix_hv_delta:+d}): "
            f"CAGR {cagr:.1%} < 10% floor. Strategy is VIX-threshold-fragile."
        )


# ── MA window sensitivity ──────────────────────────────────────────────────────

class TestMAWindowSensitivity:
    """Perturb the regime MA window by ±10 bars and check for stability."""

    @pytest.mark.slow
    @pytest.mark.parametrize("ma_delta", [-20, -10, +10, +20])
    def test_ma_window_perturbation(
        self, ma_delta, real_tqqq, real_sqqq, real_qqq, real_vix
    ):
        """±10–20 bar shift in MA window must keep CAGR above 10%."""
        tqqq = real_tqqq[real_tqqq.index >= TQQQ_INCEPTION]
        sqqq = real_sqqq[real_sqqq.index >= TQQQ_INCEPTION]
        qqq  = real_qqq[real_qqq.index   >= TQQQ_INCEPTION]

        base_ma   = REGIME_CONFIG["ma_window"]   # 150
        new_ma    = max(50, base_ma + ma_delta)  # floor at 50

        sa, sb = _make_strategies()
        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, real_vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            ma_window = new_ma,
            **{k: v for k, v in PORTFOLIO_DEFAULTS.items() if k != "ma_window"},
        )
        result = dp.run()
        cagr   = _cagr(result)

        assert cagr > 0.10, (
            f"MA window shifted by {ma_delta:+d} (→ {new_ma}): "
            f"CAGR {cagr:.1%} < 10% floor. "
            "Strategy is too sensitive to the exact MA window value."
        )
