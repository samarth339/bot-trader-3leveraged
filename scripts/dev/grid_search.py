"""
Phase 2 — Parameter Grid Search for CombinedStrategy

Tests combinations of ma_short, ma_long, mr_window, mr_band, atr_stop_mult
across real TQQQ/SQQQ data (2010–2025) and ranks by Calmar ratio.

Usage:
    python grid_search.py

Output: top 20 parameter sets sorted by Calmar ratio.
"""

import sys
import warnings
import itertools
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtester.engine import Backtester
from strategies.combined import CombinedStrategy
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION, INITIAL_CAPITAL


def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)

    vix_path = Path(f"{DATA_PROCESSED_DIR}/VIX_full.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True) if vix_path.exists() else None

    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index  >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def run_one(params, tqqq, sqqq, qqq, vix):
    strat = CombinedStrategy(**params)
    bt    = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL, vix=vix)
    result = bt.run(strat)
    m = result["metrics"]
    return {
        **params,
        "cagr":         round(m["cagr"] * 100, 2),
        "max_dd":       round(m["max_drawdown"] * 100, 2),
        "calmar":       round(m["calmar"], 3),
        "sharpe":       round(m["sharpe"], 3),
        "win_rate":     round(m["win_rate"] * 100, 1),
        "profit_factor":round(m["profit_factor"], 2),
        "n_trades":     m["n_trades"],
        "final_equity": round(m["final_equity"], 0),
    }


def main():
    print("Loading data...")
    tqqq, sqqq, qqq, vix = load_data()
    print(f"  VIX data: {'loaded' if vix is not None else 'not found — run fetch_data.py'}")

    # ── Parameter grid ────────────────────────────────────────────────────────
    grid = {
        "ma_short":      [20, 50],
        "ma_long":       [100, 200, 250],
        "roc_window":    [10, 20],
        "mr_window":     [5, 10, 20],
        "mr_band":       [0.01, 0.02, 0.03, 0.05],
        "atr_stop_mult": [2.0, 2.5, 3.5],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total  = len(combos)
    print(f"Running {total} combinations...")

    rows = []
    for idx, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        # Skip nonsensical combos
        if params["ma_short"] >= params["ma_long"]:
            continue
        try:
            row = run_one(params, tqqq, sqqq, qqq, vix)
            rows.append(row)
        except Exception:
            pass
        if idx % 50 == 0 or idx == total:
            print(f"  {idx}/{total} done  ({len(rows)} valid)")

    df = pd.DataFrame(rows)
    df = df[df["n_trades"] >= 5]   # require at least 5 trades for significance
    df = df.sort_values("calmar", ascending=False).reset_index(drop=True)

    print("\n" + "═" * 90)
    print("  TOP 20 PARAMETER SETS  (sorted by Calmar ratio)")
    print("═" * 90)
    cols = ["ma_short", "ma_long", "mr_window", "mr_band", "atr_stop_mult",
            "cagr", "max_dd", "calmar", "sharpe", "win_rate", "profit_factor", "n_trades"]
    print(df[cols].head(20).to_string(index=True))

    best = df.iloc[0]
    print(f"\nBest params:")
    for k in ["ma_short", "ma_long", "roc_window", "mr_window", "mr_band", "atr_stop_mult"]:
        print(f"  {k:20s} = {best[k]}")
    print(f"  → CAGR {best['cagr']}%  MaxDD {best['max_dd']}%  Calmar {best['calmar']}")

    out = Path("grid_search_results.csv")
    df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
