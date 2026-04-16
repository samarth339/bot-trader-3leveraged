"""
Performance reporting and stress-test analysis.
"""

import pandas as pd
import numpy as np
from config.settings import STRESS_PERIODS


def print_metrics(name: str, metrics: dict):
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Total Return    : {metrics['total_return']:>10.1%}")
    print(f"  CAGR            : {metrics['cagr']:>10.1%}")
    print(f"  Max Drawdown    : {metrics['max_drawdown']:>10.1%}")
    print(f"  Sharpe Ratio    : {metrics['sharpe']:>10.2f}")
    print(f"  Calmar Ratio    : {metrics['calmar']:>10.2f}")
    print(f"  Win Rate        : {metrics['win_rate']:>10.1%}")
    print(f"  Avg Win         : {metrics['avg_win_pct']:>10.1%}")
    print(f"  Avg Loss        : {metrics['avg_loss_pct']:>10.1%}")
    print(f"  Profit Factor   : {metrics['profit_factor']:>10.2f}")
    print(f"  # Trades        : {metrics['n_trades']:>10}")
    print(f"  Years           : {metrics['years']:>10.1f}")
    print(f"  Final Equity    : ${metrics['final_equity']:>10,.0f}")


def stress_test_report(equity_curve: pd.DataFrame, name: str = "Strategy"):
    """Print equity and drawdown stats for each crash period."""
    print(f"\n{'─'*60}")
    print(f"  Stress-Test: {name}")
    print(f"{'─'*60}")
    print(f"  {'Period':<22} {'Return':>10} {'Max DD':>10} {'End $':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")

    for label, (start, end) in STRESS_PERIODS.items():
        mask = (equity_curve.index >= start) & (equity_curve.index <= end)
        seg  = equity_curve[mask]
        if len(seg) < 2:
            print(f"  {label:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        ret = (seg["equity"].iloc[-1] / seg["equity"].iloc[0]) - 1
        dd  = seg["drawdown"].max()
        end_eq = seg["equity"].iloc[-1]
        print(f"  {label:<22} {ret:>10.1%} {dd:>10.1%} ${end_eq:>9,.0f}")


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple strategy results."""
    rows = []
    for name, r in results.items():
        m = r["metrics"].copy()
        m["name"] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("name")
    cols_pct = ["total_return", "cagr", "max_drawdown", "win_rate", "avg_win_pct", "avg_loss_pct"]
    df_display = df.copy()
    for c in cols_pct:
        if c in df_display.columns:
            df_display[c] = df_display[c].map(lambda x: f"{x:.1%}")
    for c in ["sharpe", "calmar", "profit_factor"]:
        if c in df_display.columns:
            df_display[c] = df_display[c].map(lambda x: f"{x:.2f}")
    df_display["final_equity"] = df["final_equity"].map(lambda x: f"${x:,.0f}")
    return df_display
