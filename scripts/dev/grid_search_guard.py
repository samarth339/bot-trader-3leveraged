"""
Grid Search — LongOnlyGuardStrategy

Finds the best vix_exit / vix_reentry / ma_long / confirm_bars combination
to maximize CAGR while keeping max drawdown manageable.

Usage:
    python grid_search_guard.py
"""

import sys, warnings, itertools
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtester.engine import Backtester
from strategies.long_only_guard import LongOnlyGuardStrategy
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION, INITIAL_CAPITAL


def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix_path = Path(f"{DATA_PROCESSED_DIR}/VIX_full.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True) if vix_path.exists() else None
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def run_one(params, tqqq, sqqq, qqq, vix):
    strat  = LongOnlyGuardStrategy(**params)
    bt     = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL, vix=vix)
    result = bt.run(strat)
    m = result["metrics"]
    return {
        **params,
        "cagr":          round(m["cagr"] * 100, 2),
        "max_dd":        round(m["max_drawdown"] * 100, 2),
        "calmar":        round(m["calmar"], 3),
        "sharpe":        round(m["sharpe"], 3),
        "win_rate":      round(m["win_rate"] * 100, 1),
        "profit_factor": round(m["profit_factor"], 2),
        "n_trades":      m["n_trades"],
        "final_equity":  round(m["final_equity"], 0),
    }


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()

    grid = {
        "ma_long":      [150, 200, 250],
        "vix_exit":     [22, 25, 28, 30, 35],
        "vix_reentry":  [18, 20, 22, 25],
        "confirm_bars": [1, 3, 5],
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
        except Exception as e:
            pass
        if idx % 20 == 0 or idx == total:
            print(f"  {idx}/{total}  ({len(rows)} valid)")

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)

    print("\n" + "═" * 95)
    print("  TOP 20 — sorted by CAGR")
    print("═" * 95)
    cols = ["ma_long", "vix_exit", "vix_reentry", "confirm_bars",
            "cagr", "max_dd", "calmar", "sharpe", "n_trades", "final_equity"]
    print(df[cols].head(20).to_string())

    best = df.iloc[0]
    print(f"\n✓ Best params:")
    for k in ["ma_long", "vix_exit", "vix_reentry", "confirm_bars"]:
        print(f"  {k:20s} = {best[k]}")
    print(f"  → CAGR {best['cagr']}%  MaxDD {best['max_dd']}%  "
          f"Calmar {best['calmar']}  Equity ${best['final_equity']:,.0f}")

    df.to_csv("grid_search_guard_results.csv", index=False)
    print(f"\nFull results → grid_search_guard_results.csv")


if __name__ == "__main__":
    main()
