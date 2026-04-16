"""
Grid Search — LongOnlyGuardV2

Target: CAGR >30%, Max DD 30–35%.
Tests all combinations of the 4 drawdown-reduction mechanisms.

Usage:
    python grid_search_v2.py
"""

import sys, warnings, itertools
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtester.engine import Backtester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
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


def run_one(params, tqqq, sqqq, qqq, vix):
    strat  = LongOnlyGuardV2(**params)
    bt     = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL, vix=vix)
    result = bt.run(strat)
    m = result["metrics"]
    return {
        **params,
        "cagr":         round(m["cagr"] * 100, 2),
        "max_dd":       round(m["max_drawdown"] * 100, 2),
        "calmar":       round(m["calmar"], 3),
        "sharpe":       round(m["sharpe"], 3),
        "n_trades":     m["n_trades"],
        "final_equity": round(m["final_equity"], 0),
    }


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()

    # Grid focused on hitting 30–35% DD while keeping 20%+ CAGR
    grid = {
        "ma_long":          [150, 200],
        "vix_exit":         [20, 22, 25],
        "vix_reentry":      [16, 18, 20],
        "confirm_bars":     [2, 3],
        "max_position_pct": [0.60, 0.70, 0.80],
        "vol_scale":        [True, False],
        "stagger_exit":     [True, False],
        "crash_brake_pct":  [0.20, 0.25, 0.30],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    # Filter: vix_reentry must be < vix_exit
    combos = [c for c in combos if c[2] < c[1]]
    total  = len(combos)
    print(f"Running {total} combinations...\n")

    rows = []
    for idx, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        try:
            rows.append(run_one(params, tqqq, sqqq, qqq, vix))
        except Exception:
            pass
        if idx % 100 == 0 or idx == total:
            print(f"  {idx}/{total}  ({len(rows)} valid)")

    df = pd.DataFrame(rows)

    # ── Tier 1: Sweet spot — DD ≤ 35% AND CAGR ≥ 20% ────────────────────────
    sweet = df[(df["max_dd"] <= 35) & (df["cagr"] >= 20)].sort_values("cagr", ascending=False)

    # ── Tier 2: DD ≤ 35% regardless of CAGR ─────────────────────────────────
    low_dd = df[df["max_dd"] <= 35].sort_values("cagr", ascending=False)

    # ── Tier 3: All results, sorted by Calmar ────────────────────────────────
    all_calmar = df.sort_values("calmar", ascending=False)

    print("\n" + "═" * 105)
    print("  TIER 1 — Sweet Spot: DD ≤ 35% AND CAGR ≥ 20%")
    print("═" * 105)
    cols = ["ma_long","vix_exit","vix_reentry","confirm_bars","max_position_pct",
            "vol_scale","stagger_exit","crash_brake_pct",
            "cagr","max_dd","calmar","sharpe","n_trades","final_equity"]
    if len(sweet) > 0:
        print(sweet[cols].head(20).to_string())
    else:
        print("  No combinations hit both targets. Showing best DD ≤ 35% results:")
        print(low_dd[cols].head(10).to_string())

    print("\n" + "═" * 105)
    print("  TIER 2 — All DD ≤ 35% (sorted by CAGR)")
    print("═" * 105)
    print(low_dd[cols].head(15).to_string())

    print("\n" + "═" * 105)
    print("  TIER 3 — Top 10 by Calmar (best risk-adjusted)")
    print("═" * 105)
    print(all_calmar[cols].head(10).to_string())

    df.sort_values("calmar", ascending=False).to_csv("grid_search_v2_results.csv", index=False)
    print(f"\nFull results → grid_search_v2_results.csv")
    print(f"Total valid combinations: {len(df)}")


if __name__ == "__main__":
    main()
