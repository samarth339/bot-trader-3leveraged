"""
Phase 2 Test: Adaptive VIX Thresholds vs Baseline

Compares LongOnlyGuardV2 (baseline, fixed thresholds)
against LongOnlyGuardV2Adaptive (adaptive percentile-based thresholds)
on train/val/OOS periods.

Expected impact: +0.5–1% Calmar by avoiding false exits in elevated-vol regimes.
Risk: May increase DD during sudden spikes (COVID-like).

Usage:
    python3 test_phase2_adaptive.py
    python3 test_phase2_adaptive.py --no-chart
"""

import sys, warnings
import pandas as pd
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

NO_CHART = "--no-chart" in sys.argv

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from strategies.long_only_guard_v2_adaptive import LongOnlyGuardV2Adaptive
from analysis.metrics import print_metrics
from analysis.plots import plot_equity_curves
from config.settings import INITIAL_CAPITAL, DATA_PROCESSED_DIR, TQQQ_INCEPTION
from config.strategy_config import (
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, ALLOC_CONFIG, REGIME_CONFIG
)

# Period definitions (from program.md)
PERIODS = {
    "train": ("2010-02-11", "2018-12-31"),
    "val":   ("2019-01-01", "2021-12-31"),
    "oos":   ("2022-01-01", "2025-12-31"),
}


def load_data():
    """Load all required data."""
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def slice_period(tqqq, sqqq, qqq, vix, start_date, end_date):
    """Slice data to a specific period."""
    mask = (tqqq.index >= start_date) & (tqqq.index <= end_date)
    tqqq_slice = tqqq[mask]
    sqqq_slice = sqqq[mask]
    qqq_slice = qqq[mask]
    vix_slice = vix.loc[vix.index.intersection(qqq_slice.index)]
    return tqqq_slice, sqqq_slice, qqq_slice, vix_slice


def run_dual_backtest(name, strategy_a, strategy_b, tqqq, sqqq, qqq, vix, period_label):
    """Run dual-portfolio backtest and return results."""
    dp = DualPortfolioBacktester(
        tqqq, sqqq, qqq, vix,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        initial_capital=INITIAL_CAPITAL,
        alloc_bull   = ALLOC_CONFIG["bull"],
        alloc_mid    = ALLOC_CONFIG["uncertain"],
        alloc_hi_vol = ALLOC_CONFIG["high_vol"],
        vix_bull     = REGIME_CONFIG["vix_bull"],
        vix_hi_vol   = REGIME_CONFIG["vix_hi_vol"],
        ma_window    = REGIME_CONFIG["ma_window"],
        vix_smooth   = REGIME_CONFIG["vix_smooth"],
        t1           = REGIME_CONFIG["t1_execution"],
    )
    results = dp.run()
    return results


def print_comparison(baseline_results, adaptive_results, period_label):
    """Print comparison table."""
    bm = baseline_results["metrics"]
    am = adaptive_results["metrics"]

    print(f"\n{'═'*80}")
    print(f"  {period_label.upper()} PERIOD COMPARISON")
    print(f"{'═'*80}")
    print(f"  {'Metric':<20} {'Baseline':>18} {'Adaptive':>18} {'Delta':>15}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*15}")

    metrics_to_compare = [
        ("CAGR", "cagr", lambda x: f"{x*100:.1f}%"),
        ("Max Drawdown", "max_drawdown", lambda x: f"{x*100:.1f}%"),
        ("Sharpe Ratio", "sharpe", lambda x: f"{x:.2f}"),
        ("Calmar Ratio", "calmar", lambda x: f"{x:.4f}"),
        ("Final Equity", "final_equity", lambda x: f"${x:,.0f}"),
    ]

    for label, key, fmt in metrics_to_compare:
        b_val = bm[key]
        a_val = am[key]

        # Compute delta (different logic for different metrics)
        if key == "calmar":
            delta_pct = ((a_val - b_val) / b_val * 100) if b_val != 0 else 0
            delta_str = f"{delta_pct:+.1f}%"
        elif key == "final_equity":
            delta_pct = ((a_val - b_val) / b_val * 100) if b_val != 0 else 0
            delta_str = f"{delta_pct:+.1f}%"
        elif key in ["cagr", "max_drawdown"]:
            delta_pct = (a_val - b_val) * 100
            delta_str = f"{delta_pct:+.1f}pp"
        else:
            delta_pct = a_val - b_val
            delta_str = f"{delta_pct:+.2f}"

        print(f"  {label:<20} {fmt(b_val):>18} {fmt(a_val):>18} {delta_str:>15}")


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()
    print(f"  Full history: {tqqq.index[0].date()} → {tqqq.index[-1].date()}")
    print(f"  Total bars: {len(tqqq)}\n")

    # ── Run backtest on each period ─────────────────────────────────────────────
    all_results = {}

    for period_name, (start_date, end_date) in PERIODS.items():
        print(f"\n{'─'*80}")
        print(f"  {period_name.upper()} ({start_date} → {end_date})")
        print(f"{'─'*80}")

        # Slice data
        t, s, q, v = slice_period(tqqq, sqqq, qqq, vix, start_date, end_date)
        if len(t) < 5:
            print(f"  ⚠ Not enough data for period {period_name}, skipping")
            continue

        print(f"  Bars in period: {len(t)}")

        # Create baseline strategies (fixed thresholds)
        baseline_a = LongOnlyGuardV2(
            ma_long=STRATEGY_A_CONFIG["ma_long"],
            vix_exit=STRATEGY_A_CONFIG["vix_exit"],
            vix_reentry=STRATEGY_A_CONFIG["vix_reentry"],
            confirm_bars=STRATEGY_A_CONFIG["confirm_bars"],
            max_position_pct=STRATEGY_A_CONFIG["max_position_pct"],
            vol_scale=STRATEGY_A_CONFIG["vol_scale"],
            stagger_exit=STRATEGY_A_CONFIG["stagger_exit"],
            crash_brake_pct=STRATEGY_A_CONFIG["crash_brake_pct"],
        )
        baseline_b = LongOnlyGuardV2(
            ma_long=STRATEGY_B_CONFIG["ma_long"],
            vix_exit=STRATEGY_B_CONFIG["vix_exit"],
            vix_reentry=STRATEGY_B_CONFIG["vix_reentry"],
            confirm_bars=STRATEGY_B_CONFIG["confirm_bars"],
            max_position_pct=STRATEGY_B_CONFIG["max_position_pct"],
            vol_scale=STRATEGY_B_CONFIG["vol_scale"],
            stagger_exit=STRATEGY_B_CONFIG["stagger_exit"],
            crash_brake_pct=STRATEGY_B_CONFIG["crash_brake_pct"],
        )

        # Create adaptive strategies (percentile-based thresholds)
        adaptive_a = LongOnlyGuardV2Adaptive(
            ma_long=STRATEGY_A_CONFIG["ma_long"],
            vix_exit=STRATEGY_A_CONFIG["vix_exit"],
            vix_reentry=STRATEGY_A_CONFIG["vix_reentry"],
            confirm_bars=STRATEGY_A_CONFIG["confirm_bars"],
            max_position_pct=STRATEGY_A_CONFIG["max_position_pct"],
            vol_scale=STRATEGY_A_CONFIG["vol_scale"],
            stagger_exit=STRATEGY_A_CONFIG["stagger_exit"],
            crash_brake_pct=STRATEGY_A_CONFIG["crash_brake_pct"],
            vix_percentile_adapt=True,        # ENABLED
            vix_adapt_window=252,
            vix_adapt_strength=1.0,
        )
        adaptive_b = LongOnlyGuardV2Adaptive(
            ma_long=STRATEGY_B_CONFIG["ma_long"],
            vix_exit=STRATEGY_B_CONFIG["vix_exit"],
            vix_reentry=STRATEGY_B_CONFIG["vix_reentry"],
            confirm_bars=STRATEGY_B_CONFIG["confirm_bars"],
            max_position_pct=STRATEGY_B_CONFIG["max_position_pct"],
            vol_scale=STRATEGY_B_CONFIG["vol_scale"],
            stagger_exit=STRATEGY_B_CONFIG["stagger_exit"],
            crash_brake_pct=STRATEGY_B_CONFIG["crash_brake_pct"],
            vix_percentile_adapt=True,        # ENABLED
            vix_adapt_window=252,
            vix_adapt_strength=1.0,
        )

        # Run backtests
        print(f"  Running baseline...")
        baseline_results = run_dual_backtest("Baseline", baseline_a, baseline_b, t, s, q, v, period_name)

        print(f"  Running adaptive...")
        adaptive_results = run_dual_backtest("Adaptive", adaptive_a, adaptive_b, t, s, q, v, period_name)

        # Display comparison
        print_comparison(baseline_results, adaptive_results, period_name)

        # Store for summary
        all_results[period_name] = {
            "baseline": baseline_results,
            "adaptive": adaptive_results,
        }

    # ── Summary decision ──────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  PHASE 2 DECISION")
    print(f"{'═'*80}")

    # Check if adaptive improves on train AND val without harming OOS
    train_baseline_calmar = all_results["train"]["baseline"]["metrics"]["calmar"]
    train_adaptive_calmar = all_results["train"]["adaptive"]["metrics"]["calmar"]
    train_improved = train_adaptive_calmar > train_baseline_calmar

    val_baseline_calmar = all_results["val"]["baseline"]["metrics"]["calmar"]
    val_adaptive_calmar = all_results["val"]["adaptive"]["metrics"]["calmar"]
    val_improved = val_adaptive_calmar > val_baseline_calmar

    oos_baseline_calmar = all_results["oos"]["baseline"]["metrics"]["calmar"]
    oos_adaptive_calmar = all_results["oos"]["adaptive"]["metrics"]["calmar"]
    oos_not_harmed = oos_adaptive_calmar >= oos_baseline_calmar * 0.95  # Allow 5% regression

    print(f"\n  Train Calmar:   {train_baseline_calmar:.4f} → {train_adaptive_calmar:.4f}  "
          f"({train_adaptive_calmar - train_baseline_calmar:+.4f})  {'✓' if train_improved else '✗'}")
    print(f"  Val Calmar:     {val_baseline_calmar:.4f} → {val_adaptive_calmar:.4f}  "
          f"({val_adaptive_calmar - val_baseline_calmar:+.4f})  {'✓' if val_improved else '✗'}")
    print(f"  OOS Calmar:     {oos_baseline_calmar:.4f} → {oos_adaptive_calmar:.4f}  "
          f"({oos_adaptive_calmar - oos_baseline_calmar:+.4f})  {'✓' if oos_not_harmed else '✗'}")

    print(f"\n  Acceptance Criteria:")
    print(f"    Train CAGR improves OR equal    : {'✓' if train_improved else '✗'}")
    print(f"    Val CAGR improves OR equal      : {'✓' if val_improved else '✗'}")
    print(f"    OOS not harmed (< 5% regression): {'✓' if oos_not_harmed else '✗'}")

    if train_improved and val_improved and oos_not_harmed:
        print(f"\n  ✅ RECOMMENDATION: Promote LongOnlyGuardV2Adaptive to production")
        print(f"     Adaptive VIX thresholds show promise — implement as toggleable config")
    else:
        print(f"\n  ❌ RECOMMENDATION: Keep baseline, iterate on thresholds")
        print(f"     Current adaptivity parameters may need tuning")


if __name__ == "__main__":
    main()
