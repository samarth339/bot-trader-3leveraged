"""
Robustness / Stress Test Suite — Dynamic Dual Portfolio

Tests:
  1. Regime Lag        — delay regime signal 1–3 days
  2. Regime Noise      — add ±X% noise to VIX thresholds
  3. Misclassification — randomly flip 10–20% of regime labels
  4. Transition Shock  — inject sudden VIX spikes and whipsaws
  5. Overlap Bias      — replace QQQ/VIX regime with independent SPY classifier
  6. Allocation Sweep  — vary bull/hiVol split: 80/20 → 50/50
  7. Out-of-Sample     — 2010–2018 in-sample vs 2019–2025 out-of-sample (MOST IMPORTANT)

Usage:
    python stress_test_robustness.py
"""

import sys, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

from backtester.engine import Backtester
from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION, INITIAL_CAPITAL


# ── Strategy definitions ──────────────────────────────────────────────────────

def make_best_calmar():
    return LongOnlyGuardV2(
        ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
        max_position_pct=0.90, vol_scale=False, stagger_exit=True, crash_brake_pct=0.0,
    )

def make_near_miss():
    return LongOnlyGuardV2(
        ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
        max_position_pct=0.70, vol_scale=False, stagger_exit=True, crash_brake_pct=0.30,
    )

def make_dp(tqqq, sqqq, qqq, vix,
            alloc_bull=(0.70, 0.30), alloc_mid=(0.50, 0.50), alloc_hi_vol=(0.30, 0.70),
            vix_bull=18.0, vix_hi_vol=25.0, ma_window=150):
    return DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=make_best_calmar(),
        strategy_b=make_near_miss(),
        initial_capital=INITIAL_CAPITAL,
        alloc_bull=alloc_bull, alloc_mid=alloc_mid, alloc_hi_vol=alloc_hi_vol,
        vix_bull=vix_bull, vix_hi_vol=vix_hi_vol, ma_window=ma_window,
    )


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    spy  = pd.read_csv(f"{DATA_PROCESSED_DIR}/SPY_full.csv",  index_col=0, parse_dates=True)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix, spy


def metrics_summary(results: dict) -> dict:
    m = results["metrics"]
    return {
        "cagr":   round(m["cagr"] * 100, 2),
        "max_dd": round(m["max_drawdown"] * 100, 2),
        "calmar": round(m["calmar"], 3),
        "sharpe": round(m["sharpe"], 3),
        "final":  round(m["final_equity"], 0),
    }


def print_row(label, m, baseline=None):
    delta_cagr = f"  ({m['cagr']-baseline['cagr']:+.1f}%)" if baseline else ""
    delta_dd   = f"  ({m['max_dd']-baseline['max_dd']:+.1f}%)" if baseline else ""
    print(f"  {label:<45} CAGR {m['cagr']:>5.1f}%{delta_cagr:<12}"
          f"  DD {m['max_dd']:>5.1f}%{delta_dd:<12}"
          f"  Calmar {m['calmar']:>5.3f}  Final ${m['final']:>10,.0f}")


# ── Helpers to mangle regime series ──────────────────────────────────────────

REGIMES = ["bull", "uncertain", "high_vol"]

def lag_regime(series: pd.Series, lag: int) -> pd.Series:
    """Shift regime forward by `lag` days (simulate delayed reaction)."""
    return series.shift(lag).fillna("uncertain")

def noise_regime(series: pd.Series, flip_pct: float, seed: int = 42) -> pd.Series:
    """Randomly misclassify `flip_pct` fraction of labels."""
    rng = np.random.default_rng(seed)
    s = series.copy()
    n_flip = int(len(s) * flip_pct)
    idx = rng.choice(len(s), size=n_flip, replace=False)
    for i in idx:
        others = [r for r in REGIMES if r != s.iloc[i]]
        s.iloc[i] = rng.choice(others)
    return s

def build_spy_regime(spy: pd.DataFrame, qqq_index: pd.Index,
                     ma_window: int = 200, mom_window: int = 126) -> pd.Series:
    """
    Independent regime classifier using SPY only (no VIX, different MA).
    bull     : SPY price > 200SMA  AND  SPY 6-month return > 5%
    high_vol : SPY price < 200SMA  OR   SPY 6-month return < -5%
    uncertain: everything else
    """
    spy_close = spy["close"]
    sma = spy_close.rolling(ma_window).mean()
    mom = spy_close.pct_change(mom_window)
    labels = []
    for date in qqq_index:
        if date not in spy_close.index:
            labels.append("uncertain")
            continue
        p   = spy_close.loc[date]
        s   = sma.loc[date] if not pd.isna(sma.loc[date]) else p
        m   = mom.loc[date] if not pd.isna(mom.loc[date]) else 0.0
        if p < s or m < -0.05:
            labels.append("high_vol")
        elif p > s and m > 0.05:
            labels.append("bull")
        else:
            labels.append("uncertain")
    return pd.Series(labels, index=qqq_index, name="spy_regime")


# ── Test functions ────────────────────────────────────────────────────────────

def test_baseline(tqqq, sqqq, qqq, vix):
    dp = make_dp(tqqq, sqqq, qqq, vix)
    return dp, dp.run()


def test_1_regime_lag(dp, base_regime, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 1 — Regime Classification Lag  (delayed signal 1–3 days)")
    print(f"{'═'*90}")
    print_row("Baseline (lag=0)", baseline_m)
    for lag in [1, 2, 3]:
        lagged = lag_regime(base_regime, lag)
        r = dp.run(regime_series=lagged)
        m = metrics_summary(r)
        print_row(f"Lag = {lag} day(s)", m, baseline_m)
    print("\n  Verdict: If Calmar drops > 0.20 with lag=1 → regime edge is fragile.")


def test_2_misclassification(dp, base_regime, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 2 — Regime Misclassification  (random label flips)")
    print(f"{'═'*90}")
    print_row("Baseline (0% noise)", baseline_m)
    for pct, seeds in [(0.10, [42, 7, 99]), (0.20, [42, 7, 99])]:
        cagrs, dds, calmars = [], [], []
        for seed in seeds:
            noisy = noise_regime(base_regime, pct, seed=seed)
            r = dp.run(regime_series=noisy)
            m = metrics_summary(r)
            cagrs.append(m["cagr"]); dds.append(m["max_dd"]); calmars.append(m["calmar"])
        mean_m = {"cagr": np.mean(cagrs), "max_dd": np.mean(dds),
                  "calmar": np.mean(calmars), "final": 0}
        std_cagr = np.std(cagrs)
        print_row(f"Noise {int(pct*100)}%  (mean of 3 seeds, σCAGR={std_cagr:.1f}%)", mean_m, baseline_m)
    print("\n  Verdict: If CAGR drops > 5% at 20% noise → misclassification risk is high.")


def test_3_transition_shock(tqqq, sqqq, qqq, vix, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 3 — Transition Shock  (VIX spikes and whipsaws injected into data)")
    print(f"{'═'*90}")

    vix_close = vix["close"].copy()

    # Shock A: sudden crash (bull → high_vol in 2 days) at a known calm period
    shock_a_date = pd.Timestamp("2017-08-15")   # calm summer 2017
    shock_a = vix_close.copy()
    if shock_a_date in shock_a.index:
        idx = shock_a.index.get_loc(shock_a_date)
        shock_a.iloc[idx]   = 40.0   # day 0: VIX spikes to 40
        shock_a.iloc[idx+1] = 38.0   # day 1: still high
        shock_a.iloc[idx+2] = 15.0   # day 2: snaps back

    # Shock B: whipsaw (VIX spikes then drops 5x in 10 days)
    whip_date = pd.Timestamp("2019-05-01")
    shock_b = vix_close.copy()
    if whip_date in shock_b.index:
        idx = shock_b.index.get_loc(whip_date)
        for j, val in enumerate([30, 35, 38, 35, 30, 22, 18, 15, 14, 13]):
            if idx + j < len(shock_b):
                shock_b.iloc[idx + j] = val

    for label, shocked_vix_series in [("Shock A (sudden crash 2017-08)", shock_a),
                                       ("Shock B (whipsaw 2019-05)", shock_b)]:
        vix_shocked = vix.copy()
        vix_shocked["close"] = shocked_vix_series
        dp_shocked = make_dp(tqqq, sqqq, qqq, vix_shocked)
        r = dp_shocked.run()
        m = metrics_summary(r)
        print_row(label, m, baseline_m)

    print("\n  Verdict: Small delta expected (shocks are brief). Large delta → system over-reacts.")


def test_4_overlap_bias(tqqq, sqqq, qqq, vix, spy, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 4 — Overlap Bias  (independent SPY-based regime vs QQQ/VIX regime)")
    print(f"{'═'*90}")
    print_row("Baseline (QQQ/VIX regime)", baseline_m)

    # Independent classifier: SPY 200SMA + 6-month momentum (no VIX)
    spy_regime = build_spy_regime(spy, qqq.index)
    dp = make_dp(tqqq, sqqq, qqq, vix)
    r = dp.run(regime_series=spy_regime)
    m = metrics_summary(r)
    print_row("Independent SPY regime (200SMA + 6mo momentum)", m, baseline_m)

    # Sanity: random regime (worst case)
    rng = np.random.default_rng(42)
    random_regime = pd.Series(
        rng.choice(REGIMES, size=len(qqq.index)), index=qqq.index
    )
    r_rnd = dp.run(regime_series=random_regime)
    m_rnd = metrics_summary(r_rnd)
    print_row("Random regime (sanity floor)", m_rnd, baseline_m)

    print("\n  Verdict: If SPY regime ≈ baseline → signal is robust. If ≈ random → overfit.")


def test_5_allocation_sweep(tqqq, sqqq, qqq, vix, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 5 — Allocation Sensitivity  (vary bull/hiVol weights)")
    print(f"{'═'*90}")
    print_row("Baseline (70/50/30)", baseline_m)
    configs = [
        ("80/50/20  (aggressive)", (0.80, 0.20), (0.50, 0.50), (0.20, 0.80)),
        ("70/50/30  (baseline)",   (0.70, 0.30), (0.50, 0.50), (0.30, 0.70)),
        ("60/50/40",               (0.60, 0.40), (0.50, 0.50), (0.40, 0.60)),
        ("50/50/50  (flat)",       (0.50, 0.50), (0.50, 0.50), (0.50, 0.50)),
        ("90/50/10  (max tilt)",   (0.90, 0.10), (0.50, 0.50), (0.10, 0.90)),
    ]
    for label, ab, am, ah in configs:
        dp = make_dp(tqqq, sqqq, qqq, vix, alloc_bull=ab, alloc_mid=am, alloc_hi_vol=ah)
        r = dp.run()
        m = metrics_summary(r)
        print_row(label, m, baseline_m)
    print("\n  Verdict: If results are stable across configs → allocation is not the critical variable.")


def test_6_out_of_sample(tqqq, sqqq, qqq, vix, baseline_m):
    print(f"\n{'═'*90}")
    print("  TEST 6 — OUT-OF-SAMPLE  *** MOST IMPORTANT ***")
    print(f"  Train: 2010–2018  |  Test: 2019–2025")
    print(f"{'═'*90}")

    split = pd.Timestamp("2019-01-01")

    def slice_data(df, start=None, end=None):
        if start and end:
            return df.loc[start:end]
        if start:
            return df.loc[start:]
        return df.loc[:end]

    for label, s, e in [("In-sample   2010–2018", None, "2018-12-31"),
                         ("Out-of-sample 2019–2025", "2019-01-01", None)]:
        t = slice_data(tqqq, s, e)
        sq = slice_data(sqqq, s, e)
        q = slice_data(qqq, s, e)
        v = slice_data(vix, s, e)
        if len(t) < 200:
            print(f"  {label}: insufficient data, skipping.")
            continue
        dp = make_dp(t, sq, q, v)
        r = dp.run()
        m = metrics_summary(r)
        print_row(label, m)

    print()
    print("  In-sample covers: COVID crash (2020), 2022 rate hike — in OOS period")
    print("  Out-of-sample should show degraded but still positive Calmar (target > 0.40)")
    print("\n  Verdict: OOS Calmar > 0.40 = robust.  OOS Calmar < 0.20 = overfitted.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix, spy = load_data()
    print(f"  {tqqq.index[0].date()} → {tqqq.index[-1].date()}  ({len(tqqq)} bars)\n")

    print("Running baseline...")
    dp_base, res_base = test_baseline(tqqq, sqqq, qqq, vix)
    baseline_m = metrics_summary(res_base)
    base_regime = dp_base.compute_regime_series()

    print(f"\n{'═'*90}")
    print("  BASELINE — Dynamic Dual Portfolio (70/50/30 regime-based)")
    print(f"{'═'*90}")
    print_row("Baseline", baseline_m)

    test_1_regime_lag(dp_base, base_regime, baseline_m)
    test_2_misclassification(dp_base, base_regime, baseline_m)
    test_3_transition_shock(tqqq, sqqq, qqq, vix, baseline_m)
    test_4_overlap_bias(tqqq, sqqq, qqq, vix, spy, baseline_m)
    test_5_allocation_sweep(tqqq, sqqq, qqq, vix, baseline_m)
    test_6_out_of_sample(tqqq, sqqq, qqq, vix, baseline_m)

    print(f"\n{'═'*90}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'═'*90}")
    print("""
  Test               What it reveals                      Pass threshold
  ─────────────────  ─────────────────────────────────    ─────────────────────
  1. Lag             Is timing critical?                  Calmar drop < 0.15 at lag=1
  2. Misclassify     How noisy can regime be?             CAGR drop < 5% at 20% noise
  3. Shock           Does system over-react to spikes?    DD impact < +5% per shock
  4. Overlap bias    Are QQQ+VIX signals double-counted?  SPY regime within 3% CAGR
  5. Alloc sweep     Is a specific split curve-fit?       CAGR spread < 5% across configs
  6. Out-of-sample   Does edge survive unseen data?       OOS Calmar > 0.40
""")


if __name__ == "__main__":
    main()
