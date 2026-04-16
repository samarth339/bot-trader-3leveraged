"""
Phase 2 — Strategy Development & Backtesting
─────────────────────────────────────────────
Full comparison: all strategies, 2010–2025 real ETF data.

Usage:
    python strategy_runner.py           # with charts
    python strategy_runner.py --no-chart  # metrics only, no blocking window

Pre-requisite:
    python data/fetch_data.py
"""

import sys
import pandas as pd
from pathlib import Path

# Resolve project root (scripts/dev/ → scripts/ → root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

NO_CHART = "--no-chart" in sys.argv

from backtester.engine import Backtester
from strategies.momentum_roc      import MomentumROC
from strategies.mean_reversion    import MeanReversion
from strategies.combined          import CombinedStrategy
from strategies.rsi_regime        import RSIRegimeStrategy
from strategies.dual_momentum     import DualMomentumStrategy
from strategies.trend_follow      import TrendFollowStrategy
from strategies.supertrend        import SuperTrendStrategy
from strategies.chandelier        import ChandelierStrategy
from strategies.macd_strategy     import MACDStrategy
from strategies.vix_regime        import VIXRegimeStrategy
from strategies.long_only_guard    import LongOnlyGuardStrategy
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from analysis.metrics import print_metrics, stress_test_report, compare_strategies
from analysis.plots   import plot_equity_curves, plot_trade_distribution
from config.settings  import INITIAL_CAPITAL, DATA_PROCESSED_DIR, TQQQ_INCEPTION

REAL_DATA_START = TQQQ_INCEPTION


def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)

    vix_path = Path(f"{DATA_PROCESSED_DIR}/VIX_full.csv")
    spy_path = Path(f"{DATA_PROCESSED_DIR}/SPY_full.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True) if vix_path.exists() else None
    spy = pd.read_csv(spy_path, index_col=0, parse_dates=True) if spy_path.exists() else None

    return tqqq, sqqq, qqq, vix, spy


def run_strategy(name, strategy, tqqq, sqqq, qqq, vix=None, spy=None):
    print(f"\nRunning: {name}...")
    bt = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL, vix=vix, spy=spy)
    results = bt.run(strategy)
    print_metrics(name, results["metrics"])
    stress_test_report(results["equity_curve"], name)
    return results


def main():
    print("Loading data...")
    if not Path(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv").exists():
        print("ERROR: Run:  python data/fetch_data.py")
        sys.exit(1)

    tqqq_full, sqqq_full, qqq_full, vix, spy = load_data()
    print(f"  Full history:  {qqq_full.index[0].date()} → {qqq_full.index[-1].date()}")
    print(f"  VIX: {'loaded' if vix is not None else 'missing'}  |  SPY: {'loaded' if spy is not None else 'missing'}")

    tqqq = tqqq_full[tqqq_full.index >= REAL_DATA_START]
    sqqq = sqqq_full[sqqq_full.index >= REAL_DATA_START]
    qqq  = qqq_full[qqq_full.index  >= REAL_DATA_START]
    print(f"  Backtest: {tqqq.index[0].date()} → {tqqq.index[-1].date()}  ({len(tqqq)} bars)\n")

    strategies = {
        # ── Phase 2 winner — baseline ─────────────────────────────────────────
        "★ V1 BASELINE (150MA, VIX22/20, confirm=3)": (
            LongOnlyGuardStrategy(ma_long=150, vix_exit=22, vix_reentry=20, confirm_bars=3),
            vix, None,
        ),
        # ── V3 Grid best Calmar — improved baseline ───────────────────────────
        "★ V3 BEST-CALMAR (200MA, VIX25/22, confirm=2, 90% cap, stagger)": (
            LongOnlyGuardV2(ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
                            max_position_pct=0.90, vol_scale=False,
                            stagger_exit=True, crash_brake_pct=0.0),
            vix, None,
        ),
        # ── V3 Grid near-miss — closest to 20%/35% target ────────────────────
        "★ V3 NEAR-MISS (150MA, VIX28/22, confirm=4, 70% cap, stagger, brake=30%)": (
            LongOnlyGuardV2(ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
                            max_position_pct=0.70, vol_scale=False,
                            stagger_exit=True, crash_brake_pct=0.30),
            vix, None,
        ),
        # ── V3 runner-up near-miss ────────────────────────────────────────────
        "V3 Runner-up (150MA, VIX30/22, confirm=4, 70% cap, stagger, brake=30%)": (
            LongOnlyGuardV2(ma_long=150, vix_exit=30, vix_reentry=22, confirm_bars=4,
                            max_position_pct=0.70, vol_scale=False,
                            stagger_exit=True, crash_brake_pct=0.30),
            vix, None,
        ),
        # ── V3 85% cap variant ────────────────────────────────────────────────
        "V3 85% cap (200MA, VIX25/22, confirm=2, stagger, no brake)": (
            LongOnlyGuardV2(ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
                            max_position_pct=0.85, vol_scale=False,
                            stagger_exit=True, crash_brake_pct=0.0),
            vix, None,
        ),
    }

    all_results = {}
    for name, (strat, v, s) in strategies.items():
        all_results[name] = run_strategy(name, strat, tqqq, sqqq, qqq, vix=v, spy=s)

    print(f"\nNote: {tqqq.index[0].date()} → {tqqq.index[-1].date()} (real ETF data)\n")

    print("\n" + "═" * 60)
    print("  STRATEGY COMPARISON")
    print("═" * 60)
    comparison = compare_strategies(all_results)
    # Sort by CAGR descending for max-return focus
    comparison = comparison.sort_values("cagr", ascending=False)
    print(comparison[["total_return","cagr","max_drawdown","sharpe","calmar",
                       "win_rate","profit_factor","n_trades","final_equity"]].to_string())

    best_calmar = max(all_results, key=lambda n: all_results[n]["metrics"]["calmar"])
    best_cagr   = max(all_results, key=lambda n: all_results[n]["metrics"]["cagr"])
    print(f"\nBest CAGR:   {best_cagr}")
    print(f"Best Calmar: {best_calmar}")

    if not NO_CHART:
        print("\nGenerating charts (pass --no-chart to skip)...")
        plot_equity_curves(all_results, title="Phase 2 — Max Return Strategy Search")
    else:
        print("\n(Charts skipped — run without --no-chart to view)")


if __name__ == "__main__":
    main()
