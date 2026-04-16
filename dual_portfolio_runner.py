"""
Dual-Portfolio Runner — V3 Best-Calmar + V3 Near-Miss

Dynamic allocation based on market regime:
  Strong bull  (QQQ > 150SMA, VIX < 18)  → 70% Best-Calmar / 30% Near-Miss
  Uncertain    (everything else)           → 50% / 50%
  High-vol     (VIX >= 25 or below MA)    → 30% Best-Calmar / 70% Near-Miss

Usage:
    python dual_portfolio_runner.py
    python dual_portfolio_runner.py --no-chart
"""

import sys, warnings
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

NO_CHART = "--no-chart" in sys.argv

from backtester.engine import Backtester
from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard import LongOnlyGuardStrategy
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from analysis.metrics import print_metrics, stress_test_report, compare_strategies
from analysis.plots import plot_equity_curves
from config.settings import INITIAL_CAPITAL, DATA_PROCESSED_DIR, TQQQ_INCEPTION


def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def run_single(name, strategy, tqqq, sqqq, qqq, vix):
    bt = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL, vix=vix)
    results = bt.run(strategy)
    print_metrics(name, results["metrics"])
    stress_test_report(results["equity_curve"], name)
    return results


def print_dual_metrics(name, results):
    m = results["metrics"]
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Total Return    : {m['total_return']*100:>10.1f}%")
    print(f"  CAGR            : {m['cagr']*100:>10.1f}%")
    print(f"  Max Drawdown    : {m['max_drawdown']*100:>10.1f}%")
    print(f"  Sharpe Ratio    : {m['sharpe']:>10.2f}")
    print(f"  Calmar Ratio    : {m['calmar']:>10.2f}")
    print(f"  Years           : {m['n_years']:>10.1f}")
    print(f"  Final Equity    : ${m['final_equity']:>12,.0f}")
    print(f"\n  Regime distribution:")
    rd = results["regime_distribution"]
    for regime, pct in rd.items():
        print(f"    {regime:<12} {pct:>5.1f}% of trading days")


def stress_test_dual(results, name):
    ec = results["equity_curve"]
    periods = {
        "covid_crash":    ("2020-02-19", "2020-11-30"),
        "rate_hike_2022": ("2021-11-19", "2022-12-31"),
    }
    print(f"\n{'─'*60}")
    print(f"  Stress-Test: {name}")
    print(f"{'─'*60}")
    print(f"  {'Period':<22} {'Return':>10} {'Max DD':>10} {'End $':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
    for label, (start, end) in periods.items():
        try:
            seg = ec.loc[start:end, "equity"]
            if len(seg) < 5:
                raise ValueError
            ret = (seg.iloc[-1] / seg.iloc[0]) - 1
            seg_dd = ((seg.cummax() - seg) / seg.cummax()).max()
            print(f"  {label:<22} {ret*100:>9.1f}% {seg_dd*100:>9.1f}% ${seg.iloc[-1]:>9,.0f}")
        except Exception:
            print(f"  {label:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()
    print(f"  Backtest: {tqqq.index[0].date()} → {tqqq.index[-1].date()}  ({len(tqqq)} bars)\n")

    # ── Define the two component strategies ──────────────────────────────────
    best_calmar = LongOnlyGuardV2(
        ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
        max_position_pct=0.90, vol_scale=False,
        stagger_exit=True, crash_brake_pct=0.0,
    )
    near_miss = LongOnlyGuardV2(
        ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
        max_position_pct=0.70, vol_scale=False,
        stagger_exit=True, crash_brake_pct=0.30,
    )

    # ── Run each individually for reference ──────────────────────────────────
    print("── Component strategies (individual) ──────────────────────")
    res_bc = run_single("V3 Best-Calmar  (200MA, VIX25/22, 90% cap, stagger)",
                        best_calmar, tqqq, sqqq, qqq, vix)
    res_nm = run_single("V3 Near-Miss    (150MA, VIX28/22, 70% cap, stagger, brake)",
                        near_miss,   tqqq, sqqq, qqq, vix)

    # ── Static 60/40 blend (baseline for dual portfolio) ─────────────────────
    print("\n── Static 60/40 blend ──────────────────────────────────────")
    dp_static = DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=LongOnlyGuardV2(
            ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
            max_position_pct=0.90, vol_scale=False,
            stagger_exit=True, crash_brake_pct=0.0,
        ),
        strategy_b=LongOnlyGuardV2(
            ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
            max_position_pct=0.70, vol_scale=False,
            stagger_exit=True, crash_brake_pct=0.30,
        ),
        initial_capital=INITIAL_CAPITAL,
        alloc_bull   = (0.60, 0.40),
        alloc_mid    = (0.60, 0.40),
        alloc_hi_vol = (0.60, 0.40),
    )
    res_static = dp_static.run()
    print_dual_metrics("Static 60/40  (Best-Calmar 60%, Near-Miss 40%)", res_static)
    stress_test_dual(res_static, "Static 60/40")

    # ── Dynamic allocation ────────────────────────────────────────────────────
    print("\n── Dynamic allocation (regime-based) ───────────────────────")
    dp_dynamic = DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=LongOnlyGuardV2(
            ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
            max_position_pct=0.90, vol_scale=False,
            stagger_exit=True, crash_brake_pct=0.0,
        ),
        strategy_b=LongOnlyGuardV2(
            ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
            max_position_pct=0.70, vol_scale=False,
            stagger_exit=True, crash_brake_pct=0.30,
        ),
        initial_capital=INITIAL_CAPITAL,
        alloc_bull   = (0.70, 0.30),   # strong bull  → 70% Best-Calmar
        alloc_mid    = (0.50, 0.50),   # uncertain    → 50/50
        alloc_hi_vol = (0.30, 0.70),   # high-vol     → 70% Near-Miss
        vix_bull     = 18.0,
        vix_hi_vol   = 25.0,
        ma_window    = 150,
    )
    res_dynamic = dp_dynamic.run()
    print_dual_metrics("Dynamic Allocation (70/50/30 regime-based)", res_dynamic)
    stress_test_dual(res_dynamic, "Dynamic Allocation")

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  SUMMARY COMPARISON")
    print(f"{'═'*72}")
    rows = [
        ("V3 Best-Calmar (solo)",    res_bc["metrics"]),
        ("V3 Near-Miss (solo)",      res_nm["metrics"]),
        ("Static 60/40 blend",       res_static["metrics"]),
        ("Dynamic allocation",       res_dynamic["metrics"]),
    ]
    print(f"  {'Strategy':<30} {'CAGR':>7} {'Max DD':>8} {'Calmar':>8} {'Final $':>12}")
    print(f"  {'─'*30} {'─'*7} {'─'*8} {'─'*8} {'─'*12}")
    for name, m in rows:
        print(f"  {name:<30} {m['cagr']*100:>6.1f}% {m['max_drawdown']*100:>7.1f}% "
              f"{m['calmar']:>8.2f} ${m['final_equity']:>11,.0f}")

    if not NO_CHART:
        # Build equity curves dict for plotting
        all_results = {
            "V3 Best-Calmar":    res_bc,
            "V3 Near-Miss":      res_nm,
            "Static 60/40":      {"equity_curve": res_static["equity_curve"],
                                  "metrics": res_static["metrics"], "trades": pd.DataFrame()},
            "Dynamic Alloc":     {"equity_curve": res_dynamic["equity_curve"],
                                  "metrics": res_dynamic["metrics"], "trades": pd.DataFrame()},
        }
        plot_equity_curves(all_results, title="Dual-Portfolio — Dynamic Allocation")
    else:
        print("\n(Charts skipped — run without --no-chart to view)")


if __name__ == "__main__":
    main()
