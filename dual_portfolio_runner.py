"""
Dual-Portfolio Runner — LOCKED production configuration

ALL strategy parameters and allocations come from config/strategy_config.py
(single source of truth). Nothing is hardcoded here — if the locked config
changes, this runner reflects it automatically.

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
from backtester.exposure_replay import build_strategy
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from analysis.metrics import print_metrics, stress_test_report, compare_strategies
from analysis.plots import plot_equity_curves
from config.settings import INITIAL_CAPITAL, DATA_PROCESSED_DIR, TQQQ_INCEPTION
from config.strategy_config import (
    PORTFOLIO_DEFAULTS, STRATEGY_A_CONFIG, STRATEGY_B_CONFIG,
)


def build_locked_components(initial_capital: float = INITIAL_CAPITAL,
                            data: tuple = None) -> DualPortfolioBacktester:
    """
    Construct the DualPortfolioBacktester exactly as locked in
    config/strategy_config.py. Importable so tests can assert that this
    runner cannot drift from the production configuration.
    """
    if data is None:
        data = load_data()
    tqqq, sqqq, qqq, vix = data
    return DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=build_strategy(STRATEGY_A_CONFIG),
        strategy_b=build_strategy(STRATEGY_B_CONFIG),
        initial_capital=initial_capital,
        alloc_bull   = PORTFOLIO_DEFAULTS["alloc_bull"],
        alloc_mid    = PORTFOLIO_DEFAULTS["alloc_mid"],
        alloc_hi_vol = PORTFOLIO_DEFAULTS["alloc_hi_vol"],
        vix_bull     = PORTFOLIO_DEFAULTS["vix_bull"],
        vix_hi_vol   = PORTFOLIO_DEFAULTS["vix_hi_vol"],
        ma_window    = PORTFOLIO_DEFAULTS["ma_window"],
        t1           = PORTFOLIO_DEFAULTS["t1"],
        confirm_days = PORTFOLIO_DEFAULTS["confirm_days"],
        vix_smooth   = PORTFOLIO_DEFAULTS["vix_smooth"],
    )


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

    # ── Component strategies from the LOCKED config ──────────────────────────
    a_cfg, b_cfg = STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
    label_a = (f"Strategy A {a_cfg['name']}  ({a_cfg['ma_long']}MA, "
               f"VIX{a_cfg['vix_exit']}/{a_cfg['vix_reentry']}, "
               f"{a_cfg['max_position_pct']:.0%} cap)")
    label_b = (f"Strategy B {b_cfg['name']}  ({b_cfg['ma_long']}MA, "
               f"VIX{b_cfg['vix_exit']}/{b_cfg['vix_reentry']}, "
               f"{b_cfg['max_position_pct']:.0%} cap)")

    # ── Run each individually for reference ──────────────────────────────────
    print("── Component strategies (individual, locked config) ────────")
    res_a = run_single(label_a, build_strategy(a_cfg), tqqq, sqqq, qqq, vix)
    res_b = run_single(label_b, build_strategy(b_cfg), tqqq, sqqq, qqq, vix)

    # ── Static 60/40 blend (comparison baseline) ─────────────────────────────
    print("\n── Static 60/40 blend (baseline) ───────────────────────────")
    dp_static = build_locked_components(data=(tqqq, sqqq, qqq, vix))
    dp_static.alloc_bull = dp_static.alloc_mid = dp_static.alloc_hi_vol = (0.60, 0.40)
    res_static = dp_static.run()
    print_dual_metrics("Static 60/40  (Strategy A 60%, Strategy B 40%)", res_static)
    stress_test_dual(res_static, "Static 60/40")

    # ── Dynamic allocation — THE locked production configuration ─────────────
    print("\n── Dynamic allocation (LOCKED production config) ───────────")
    dp_dynamic = build_locked_components(data=(tqqq, sqqq, qqq, vix))
    res_dynamic = dp_dynamic.run()
    ab = PORTFOLIO_DEFAULTS["alloc_bull"][0]
    am = PORTFOLIO_DEFAULTS["alloc_mid"][0]
    ah = PORTFOLIO_DEFAULTS["alloc_hi_vol"][0]
    print_dual_metrics(
        f"Dynamic Allocation ({ab:.0%}/{am:.0%}/{ah:.0%} regime-based, locked)",
        res_dynamic)
    stress_test_dual(res_dynamic, "Dynamic Allocation (locked)")

    # ── Summary comparison ────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  SUMMARY COMPARISON")
    print(f"{'═'*72}")
    rows = [
        ("Strategy A (solo)",            res_a["metrics"]),
        ("Strategy B (solo)",            res_b["metrics"]),
        ("Static 60/40 blend",           res_static["metrics"]),
        ("Dynamic allocation (locked)",  res_dynamic["metrics"]),
    ]
    print(f"  {'Strategy':<30} {'CAGR':>7} {'Max DD':>8} {'Calmar':>8} {'Final $':>12}")
    print(f"  {'─'*30} {'─'*7} {'─'*8} {'─'*8} {'─'*12}")
    for name, m in rows:
        print(f"  {name:<30} {m['cagr']*100:>6.1f}% {m['max_drawdown']*100:>7.1f}% "
              f"{m['calmar']:>8.2f} ${m['final_equity']:>11,.0f}")

    if not NO_CHART:
        # Build equity curves dict for plotting
        all_results = {
            "Strategy A":        res_a,
            "Strategy B":        res_b,
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
