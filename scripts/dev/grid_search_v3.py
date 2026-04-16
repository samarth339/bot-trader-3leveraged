"""
Grid Search V3 — LongOnlyGuardV2  (expanded, targeting >30% CAGR with DD 30–35%)

The V2 grid showed no combo hits DD≤35% AND CAGR≥20%.
This search uses looser / wider parameters to find the knee of the curve:
  - vix_exit up to 30 (later exit = more time in TQQQ)
  - max_position_pct up to 0.90
  - crash_brake_pct up to 0.40
  - vol_scale=False focus (vol_scale consistently kills CAGR)
  - stagger_exit=True focus (best Tier 3 winner used this)
  - Mech1-only variants (no vol_scale, no crash_brake) to isolate cap effect

Usage:
    python grid_search_v3.py
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

    # Expanded grid — focus on higher cap + looser VIX exit to preserve CAGR
    grid = {
        "ma_long":          [150, 200],
        "vix_exit":         [22, 25, 28, 30],
        "vix_reentry":      [18, 20, 22],
        "confirm_bars":     [2, 3, 4],
        "max_position_pct": [0.70, 0.80, 0.85, 0.90],
        "vol_scale":        [False],          # vol_scale=True consistently kills CAGR
        "stagger_exit":     [True, False],
        "crash_brake_pct":  [0.0, 0.30, 0.40],   # 0 = disabled
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

    cols = ["ma_long","vix_exit","vix_reentry","confirm_bars","max_position_pct",
            "vol_scale","stagger_exit","crash_brake_pct",
            "cagr","max_dd","calmar","sharpe","n_trades","final_equity"]

    # ── Tier 1: True sweet spot ───────────────────────────────────────────────
    sweet = df[(df["max_dd"] <= 35) & (df["cagr"] >= 20)].sort_values("cagr", ascending=False)

    # ── Tier 2: DD ≤ 40% AND CAGR ≥ 18% ─────────────────────────────────────
    near  = df[(df["max_dd"] <= 40) & (df["cagr"] >= 18)].sort_values("cagr", ascending=False)

    # ── Tier 3: DD ≤ 35% sorted by CAGR ─────────────────────────────────────
    low_dd = df[df["max_dd"] <= 35].sort_values("cagr", ascending=False)

    # ── Tier 4: All results by Calmar ─────────────────────────────────────────
    all_calmar = df.sort_values("calmar", ascending=False)

    print("\n" + "═" * 110)
    print("  TIER 1 — Sweet Spot: DD ≤ 35% AND CAGR ≥ 20%")
    print("═" * 110)
    if len(sweet) > 0:
        print(sweet[cols].head(20).to_string())
    else:
        print("  No combinations hit both targets.")

    print("\n" + "═" * 110)
    print("  TIER 2 — Near Miss: DD ≤ 40% AND CAGR ≥ 18%")
    print("═" * 110)
    if len(near) > 0:
        print(near[cols].head(20).to_string())
    else:
        print("  No combinations found.")

    print("\n" + "═" * 110)
    print("  TIER 3 — All DD ≤ 35% (sorted by CAGR)")
    print("═" * 110)
    print(low_dd[cols].head(15).to_string())

    print("\n" + "═" * 110)
    print("  TIER 4 — Top 10 by Calmar (best risk-adjusted, any DD)")
    print("═" * 110)
    print(all_calmar[cols].head(10).to_string())

    df.sort_values("calmar", ascending=False).to_csv("grid_search_v3_results.csv", index=False)
    print(f"\nFull results → grid_search_v3_results.csv")
    print(f"Total valid combinations: {len(df)}")


if __name__ == "__main__":
    main()
