"""
Optimization Runner — Steps 1–4

Step 1: T-1 Execution Model    (honest production baseline)
Step 2: Regime Stabilizer      (confirm_days=2, vix_smooth=3)
Step 3: Bull Allocation Sweep  (70/30 → 85/15 in bull regime only)
Step 4: Confidence-Weighted    (continuous, no step function)

Usage:
    python optimize_dual.py
    python optimize_dual.py --no-chart
"""

import sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")

NO_CHART = "--no-chart" in sys.argv

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from analysis.plots import plot_equity_curves
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION, INITIAL_CAPITAL


def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def make_dp(tqqq, sqqq, qqq, vix,
            alloc_bull=(0.70, 0.30), alloc_mid=(0.50, 0.50), alloc_hi_vol=(0.30, 0.70),
            t1=False, confirm_days=1, vix_smooth=1,
            w_a_min=0.15, w_a_max=0.85):
    return DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=LongOnlyGuardV2(
            ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
            max_position_pct=0.90, vol_scale=False, stagger_exit=True, crash_brake_pct=0.0,
        ),
        strategy_b=LongOnlyGuardV2(
            ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
            max_position_pct=0.70, vol_scale=False, stagger_exit=True, crash_brake_pct=0.30,
        ),
        initial_capital=INITIAL_CAPITAL,
        alloc_bull=alloc_bull, alloc_mid=alloc_mid, alloc_hi_vol=alloc_hi_vol,
        t1=t1, confirm_days=confirm_days, vix_smooth=vix_smooth,
        w_a_min=w_a_min, w_a_max=w_a_max,
    )


def fmt(m, baseline=None):
    d_cagr = f" ({m['cagr']-baseline['cagr']:+.1f}%)" if baseline else ""
    d_dd   = f" ({m['max_dd']-baseline['max_dd']:+.1f}%)" if baseline else ""
    d_cal  = f" ({m['calmar']-baseline['calmar']:+.3f})" if baseline else ""
    return (f"  CAGR {m['cagr']:>5.1f}%{d_cagr:<10}"
            f"  DD {m['max_dd']:>5.1f}%{d_dd:<10}"
            f"  Calmar {m['calmar']:>5.3f}{d_cal:<10}"
            f"  Final ${m['final']:>11,.0f}")


def run_and_summarise(dp, label, baseline=None, confidence=False):
    r = dp.run_confidence_weighted() if confidence else dp.run()
    m = {
        "cagr":   round(r["metrics"]["cagr"] * 100, 2),
        "max_dd": round(r["metrics"]["max_drawdown"] * 100, 2),
        "calmar": round(r["metrics"]["calmar"], 3),
        "sharpe": round(r["metrics"]["sharpe"], 3),
        "final":  round(r["metrics"]["final_equity"], 0),
    }
    print(f"  {label:<52}{fmt(m, baseline)}")
    return r, m


def stress_row(results, label, period_start, period_end):
    ec = results["equity_curve"]["equity"]
    try:
        seg = ec.loc[period_start:period_end]
        if len(seg) < 5:
            raise ValueError
        ret = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        dd  = ((seg.cummax() - seg) / seg.cummax()).max() * 100
        print(f"    {label:<20} Return {ret:>7.1f}%   Max DD {dd:>6.1f}%   End ${seg.iloc[-1]:>10,.0f}")
    except Exception:
        print(f"    {label:<20} N/A")


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()
    print(f"  {tqqq.index[0].date()} → {tqqq.index[-1].date()}  ({len(tqqq)} bars)\n")

    divider = "═" * 100

    # ─────────────────────────────────────────────────────────────────────────
    # Original backtest baseline (same-bar, raw VIX, no confirmation)
    # ─────────────────────────────────────────────────────────────────────────
    print(divider)
    print("  ORIGINAL BASELINE  (same-bar regime, raw VIX, no confirmation)")
    print(divider)
    dp0 = make_dp(tqqq, sqqq, qqq, vix)
    r0, base = run_and_summarise(dp0, "Original baseline (T+0, no stabiliser)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: T-1 Execution Model
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  STEP 1 — T-1 Execution Model  (honest production baseline)")
    print(divider)
    dp1 = make_dp(tqqq, sqqq, qqq, vix, t1=True)
    r1, m1 = run_and_summarise(dp1, "T-1 regime (production-honest)", base)
    print(f"\n  → This is the number that matters. T+0 is not achievable in live trading.")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Regime Stabiliser
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  STEP 2 — Regime Stabiliser  (reduce twitchiness)")
    print(f"  T-1 is always ON from here. Baseline = T-1 result above.")
    print(divider)
    configs = [
        ("T-1 only (no stabiliser)",          dict(t1=True)),
        ("T-1 + confirm=2 days",              dict(t1=True, confirm_days=2)),
        ("T-1 + confirm=3 days",              dict(t1=True, confirm_days=3)),
        ("T-1 + VIX smooth=3d",               dict(t1=True, vix_smooth=3)),
        ("T-1 + VIX smooth=5d",               dict(t1=True, vix_smooth=5)),
        ("T-1 + confirm=2 + VIX smooth=3d",   dict(t1=True, confirm_days=2, vix_smooth=3)),
        ("T-1 + confirm=2 + VIX smooth=5d",   dict(t1=True, confirm_days=2, vix_smooth=5)),
    ]
    best_stab_label, best_stab_m, best_stab_dp = None, None, None
    for label, kwargs in configs:
        dp = make_dp(tqqq, sqqq, qqq, vix, **kwargs)
        r, m = run_and_summarise(dp, label, m1)
        if best_stab_m is None or m["calmar"] > best_stab_m["calmar"]:
            best_stab_label, best_stab_m, best_stab_dp = label, m, dp
    print(f"\n  → Best stabiliser: {best_stab_label}  (Calmar {best_stab_m['calmar']:.3f})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Bull Allocation Sweep (T-1 + best stabiliser)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  STEP 3 — Bull Allocation Sweep  (vary bull-regime weight only)")
    print(f"  Building on: {best_stab_label}")
    print(divider)
    # Extract best stabiliser kwargs
    stab_kwargs = dict(t1=True,
                       confirm_days=best_stab_dp.confirm_days,
                       vix_smooth=best_stab_dp.vix_smooth)

    bull_configs = [
        ("70/30 bull (original)",  (0.70, 0.30)),
        ("75/25 bull",             (0.75, 0.25)),
        ("80/20 bull",             (0.80, 0.20)),
        ("82/18 bull",             (0.82, 0.18)),
        ("85/15 bull",             (0.85, 0.15)),
        ("88/12 bull",             (0.88, 0.12)),
        ("90/10 bull",             (0.90, 0.10)),
    ]
    best_bull_label, best_bull_m, best_bull_dp = None, None, None
    for label, ab in bull_configs:
        dp = make_dp(tqqq, sqqq, qqq, vix, alloc_bull=ab, **stab_kwargs)
        r, m = run_and_summarise(dp, label, best_stab_m)
        if best_bull_m is None or m["calmar"] > best_bull_m["calmar"]:
            best_bull_label, best_bull_m, best_bull_dp = label, m, dp
    print(f"\n  → Best bull split: {best_bull_label}  (Calmar {best_bull_m['calmar']:.3f})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Confidence-Weighted Allocation
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  STEP 4 — Confidence-Weighted Allocation  (continuous, no step function)")
    print(f"  Building on: {best_stab_label}")
    print(divider)
    cw_configs = [
        ("CW 15→85  (default range)",  0.15, 0.85),
        ("CW 20→85",                   0.20, 0.85),
        ("CW 20→80",                   0.20, 0.80),
        ("CW 25→85",                   0.25, 0.85),
        ("CW 30→80",                   0.30, 0.80),
        ("CW 30→85",                   0.30, 0.85),
    ]
    best_cw_label, best_cw_m, best_cw_r = None, None, None
    for label, w_min, w_max in cw_configs:
        dp = make_dp(tqqq, sqqq, qqq, vix, w_a_min=w_min, w_a_max=w_max, **stab_kwargs)
        r, m = run_and_summarise(dp, label, best_stab_m, confidence=True)
        if best_cw_m is None or m["calmar"] > best_cw_m["calmar"]:
            best_cw_label, best_cw_m, best_cw_r = label, m, r
    print(f"\n  → Best confidence-weighted: {best_cw_label}  (Calmar {best_cw_m['calmar']:.3f})")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL COMPARISON + STRESS TESTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  FINAL COMPARISON — All Steps  (stress-tested)")
    print(divider)

    candidates = [
        ("1. Original (T+0)",               r0),
        ("2. T-1 honest baseline",          r1),
        (f"3. T-1 + Best stabiliser ({best_stab_label.split()[2:4]})", best_stab_dp.run()),
        (f"4. T-1 + Best bull split ({best_bull_label})",              best_bull_dp.run()),
        (f"5. Confidence-weighted ({best_cw_label})",                  best_cw_r),
    ]

    col_w = 52
    print(f"\n  {'Strategy':<{col_w}} {'CAGR':>7} {'Max DD':>8} {'Calmar':>8} {'Sharpe':>7} {'Final $':>13}")
    print(f"  {'─'*col_w} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*13}")
    for name, r in candidates:
        m = r["metrics"]
        print(f"  {name:<{col_w}} "
              f"{m['cagr']*100:>6.1f}%  "
              f"{m['max_drawdown']*100:>7.1f}%  "
              f"{m['calmar']:>8.3f}  "
              f"{m['sharpe']:>7.3f}  "
              f"${m['final_equity']:>12,.0f}")

    print(f"\n  Stress tests:")
    for name, r in candidates:
        print(f"\n    {name}")
        stress_row(r, "COVID 2020",   "2020-02-19", "2020-11-30")
        stress_row(r, "Rates 2022",   "2021-11-19", "2022-12-31")

    if not NO_CHART:
        ec_dict = {name: {"equity_curve": r["equity_curve"],
                          "metrics": r["metrics"], "trades": __import__("pandas").DataFrame()}
                   for name, r in candidates}
        plot_equity_curves(ec_dict, title="Dual Portfolio — Step-by-Step Optimization")
    else:
        print("\n(Charts skipped — run without --no-chart to view)")


if __name__ == "__main__":
    main()
