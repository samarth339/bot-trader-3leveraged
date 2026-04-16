"""
Autoresearch Evaluation Harness
================================
Runs the dual-portfolio backtest on train and val periods,
returns the composite score used by the agent loop.

Usage (standalone):
    python autoresearch/run_backtest.py
    python autoresearch/run_backtest.py --oos        # also report OOS period
    python autoresearch/run_backtest.py --json       # output JSON instead of table

This is the equivalent of Karpathy's 5-minute training run + val_bpb evaluation.
Each call takes ~30 seconds (full dual-portfolio backtest on 8+ years of data).
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.strategy_config import (
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG,
    REGIME_CONFIG, ALLOC_CONFIG, PORTFOLIO_DEFAULTS,
)
from config.settings import INITIAL_CAPITAL

# ── Evaluation Periods (NEVER CHANGE) ─────────────────────────────────────────
PERIODS = {
    "train": ("2010-02-11", "2018-12-31"),
    "val":   ("2019-01-01", "2021-12-31"),
    "oos":   ("2022-01-01", "2025-12-31"),
}

# ── Veto Constraints ───────────────────────────────────────────────────────────
VETO_TRAIN = {
    "cagr":       (">=", 0.24),
    "max_drawdown": ("<=", 0.42),
    "sharpe":     (">=", 0.80),
}
VETO_VAL = {
    # Val period 2019–2021 covers COVID crash (March 2020 VIX=89).
    # Baseline config scores: CAGR=18%, MaxDD=55.5%, Calmar=0.325.
    # Veto thresholds are set BELOW baseline so the agent has room to improve,
    # but above catastrophic failure levels.
    "cagr":         (">=", 0.12),   # baseline 18% — floor at 12%
    "max_drawdown": ("<=", 0.65),   # baseline 55.5% — ceiling at 65%
    "calmar":       (">=", 0.20),   # baseline 0.325 — floor at 0.20
}

# ── Data ───────────────────────────────────────────────────────────────────────
_DATA_DIR = _ROOT / "data" / "processed"


def _load_data():
    """Load all four data CSVs. Raises FileNotFoundError if missing."""
    files = {
        "tqqq": _DATA_DIR / "TQQQ_full.csv",
        "sqqq": _DATA_DIR / "SQQQ_full.csv",
        "qqq":  _DATA_DIR / "QQQ_full.csv",
        "vix":  _DATA_DIR / "VIX_full.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")
    return {k: pd.read_csv(v, index_col=0, parse_dates=True) for k, v in files.items()}


def _slice_period(data: dict, start: str, end: str) -> dict:
    """Slice all four DataFrames to a date range and align to common dates."""
    sliced = {k: df.loc[start:end] for k, df in data.items()}
    common = (sliced["tqqq"].index
              .intersection(sliced["sqqq"].index)
              .intersection(sliced["qqq"].index)
              .intersection(sliced["vix"].index))
    return {k: df.loc[common] for k, df in sliced.items()}


# ── Backtest Runner ────────────────────────────────────────────────────────────

def run_period(data_slice: dict, period_name: str) -> dict:
    """
    Run the dual-portfolio backtest on one slice of data.
    Returns metrics dict with keys: cagr, max_drawdown, calmar, sharpe, final_equity.
    """
    sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
    sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})

    dp = DualPortfolioBacktester(
        data_slice["tqqq"], data_slice["sqqq"],
        data_slice["qqq"],  data_slice["vix"],
        strategy_a=sa, strategy_b=sb,
        initial_capital=INITIAL_CAPITAL,
        **PORTFOLIO_DEFAULTS,
    )
    result = dp.run()
    m = result["metrics"]
    return {
        "period":       period_name,
        "cagr":         round(m["cagr"], 4),
        "max_drawdown": round(m["max_drawdown"], 4),
        "calmar":       round(m["calmar"], 4),
        "sharpe":       round(m["sharpe"], 4),
        "final_equity": round(m["final_equity"], 2),
        "n_bars":       len(data_slice["tqqq"]),
    }


def check_veto(metrics: dict, constraints: dict) -> tuple[bool, list[str]]:
    """
    Check hard veto constraints.
    Returns (passed: bool, violations: list[str]).
    """
    violations = []
    for key, (op, threshold) in constraints.items():
        val = metrics.get(key, 0.0)
        if op == ">=" and val < threshold:
            violations.append(f"{key}={val:.4f} < {threshold} (veto)")
        elif op == "<=" and val > threshold:
            violations.append(f"{key}={val:.4f} > {threshold} (veto)")
    return len(violations) == 0, violations


def composite_score(train: dict, val: dict) -> float:
    """
    Geometric mean of train Calmar × val Calmar.
    Returns 0.0 if any veto constraint is violated.
    """
    train_ok, train_viols = check_veto(train, VETO_TRAIN)
    val_ok,   val_viols   = check_veto(val,   VETO_VAL)

    if not train_ok or not val_ok:
        return 0.0

    return float((train["calmar"] * val["calmar"]) ** 0.5)


# ── Main Entry Point ────────────────────────────────────────────────────────────

def evaluate(include_oos: bool = False) -> dict:
    """
    Full evaluation: train + val (+ optional OOS).
    Returns dict with all period metrics and the composite score.
    """
    data = _load_data()

    train_slice = _slice_period(data, *PERIODS["train"])
    val_slice   = _slice_period(data, *PERIODS["val"])

    train_metrics = run_period(train_slice, "train")
    val_metrics   = run_period(val_slice,   "val")
    score         = composite_score(train_metrics, val_metrics)

    train_ok, train_viols = check_veto(train_metrics, VETO_TRAIN)
    val_ok,   val_viols   = check_veto(val_metrics,   VETO_VAL)

    result = {
        "composite_score": round(score, 6),
        "train":           train_metrics,
        "val":             val_metrics,
        "veto_violations": train_viols + val_viols,
        "accepted":        score > 0.0,
    }

    if include_oos:
        oos_slice = _slice_period(data, *PERIODS["oos"])
        result["oos"] = run_period(oos_slice, "oos")

    return result


def print_result(result: dict):
    """Human-readable output."""
    score   = result["composite_score"]
    train   = result["train"]
    val     = result["val"]
    verdict = "ACCEPT" if result["accepted"] else "REJECT"

    print(f"\n{'═'*62}")
    print(f"  COMPOSITE SCORE: {score:.4f}   [{verdict}]")
    print(f"{'═'*62}")
    print(f"  {'Period':<10} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Sharpe':>8}")
    print(f"  {'─'*50}")
    for m in [train, val] + ([result["oos"]] if "oos" in result else []):
        print(
            f"  {m['period']:<10}"
            f"  {m['cagr']*100:>6.1f}%"
            f"  {m['max_drawdown']*100:>6.1f}%"
            f"  {m['calmar']:>8.3f}"
            f"  {m['sharpe']:>8.3f}"
        )
    if result["veto_violations"]:
        print(f"\n  VETO VIOLATIONS:")
        for v in result["veto_violations"]:
            print(f"    ✗ {v}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos",  action="store_true", help="Also evaluate OOS period")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    result = evaluate(include_oos=args.oos)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)
