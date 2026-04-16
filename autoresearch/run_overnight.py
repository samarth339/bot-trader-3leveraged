"""
Overnight Runner
================
Wrapper around agent_loop.py that:
  - Validates environment before starting
  - Runs the full experiment session
  - Sends email summary on completion
  - Prints next-morning review instructions

Usage:
    python3 autoresearch/run_overnight.py             # 300 experiments (~8 hrs)
    python3 autoresearch/run_overnight.py -n 50       # quick test run
    python3 autoresearch/run_overnight.py --fast      # Haiku model, faster/cheaper
"""

from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

_AR = Path(__file__).parent


def _check_env() -> list:
    """Return list of problems found before starting."""
    problems = []

    # API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY not set (add to .env or export in shell)")

    # anthropic package
    try:
        import anthropic
    except ImportError:
        problems.append("anthropic package missing — run: pip3 install anthropic")

    # Data files
    data_dir = _ROOT / "data" / "processed"
    for f in ["TQQQ_full.csv", "SQQQ_full.csv", "QQQ_full.csv", "VIX_full.csv"]:
        if not (data_dir / f).exists():
            problems.append(f"Missing data file: {data_dir / f}")

    # Config file
    if not (_ROOT / "config" / "strategy_config.py").exists():
        problems.append("config/strategy_config.py missing")

    # run_backtest.py sanity
    if not (_AR / "run_backtest.py").exists():
        problems.append("autoresearch/run_backtest.py missing")

    return problems


def _estimate_time(n_experiments: int) -> str:
    secs = n_experiments * 40   # ~40s per experiment
    h, m = divmod(secs, 3600)
    m //= 60
    return f"~{h}h {m:02d}m"


def _print_banner(n_exp: int, model: str) -> None:
    est = _estimate_time(n_exp)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          AUTORESEARCH OVERNIGHT RUNNER                   ║
╠══════════════════════════════════════════════════════════╣
║  Experiments : {n_exp:<10}  Estimated: {est:<14}      ║
║  Model       : {model:<42} ║
║  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M'):<42} ║
║  Log file    : autoresearch/run.log                      ║
╠══════════════════════════════════════════════════════════╣
║  Press Ctrl+C to stop cleanly at any time.               ║
║  Config will be auto-restored to last winning state.     ║
╚══════════════════════════════════════════════════════════╝
""")


def run(n_experiments: int = 300, model: str = "claude-sonnet-4-5",
        resume: bool = False) -> None:
    # ── Pre-flight ──────────────────────────────────────────────────────────
    print("Checking environment...")
    problems = _check_env()
    if problems:
        print("\n❌  Cannot start — fix these issues first:\n")
        for p in problems:
            print(f"  • {p}")
        print()
        sys.exit(1)

    print("✓  All checks passed.\n")

    # ── Quick sanity: run one backtest to verify baseline works ────────────
    print("Running baseline backtest to verify setup...")
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(_AR / "run_backtest.py"), "--json"],
        capture_output=True, text=True, timeout=180, cwd=str(_ROOT),
    )
    if proc.returncode != 0:
        print("❌  Baseline backtest failed:")
        print(proc.stderr[-400:])
        sys.exit(1)

    import json
    baseline = json.loads(proc.stdout)
    print(f"✓  Baseline score: {baseline['composite_score']:.5f}")
    print(f"   Train: CAGR {baseline['train']['cagr']*100:.1f}%  "
          f"MaxDD {baseline['train']['max_drawdown']*100:.1f}%  "
          f"Calmar {baseline['train']['calmar']:.3f}")
    print(f"   Val  : CAGR {baseline['val']['cagr']*100:.1f}%  "
          f"MaxDD {baseline['val']['max_drawdown']*100:.1f}%  "
          f"Calmar {baseline['val']['calmar']:.3f}")
    print()

    _print_banner(n_experiments, model)
    time.sleep(2)

    # ── Launch loop ────────────────────────────────────────────────────────
    from autoresearch.agent_loop import run_loop
    run_loop(
        max_experiments=n_experiments,
        model=model,
        dry_run=False,
        resume=resume,
    )

    # ── Morning instructions ───────────────────────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────┐
│  MORNING REVIEW — run these commands:                    │
│                                                          │
│  # 1. See all wins and their scores                      │
│  python3 autoresearch/review_results.py --wins           │
│                                                          │
│  # 2. Evaluate the best config on all periods incl. OOS  │
│  python3 autoresearch/run_backtest.py --oos              │
│                                                          │
│  # 3. Run the full test suite on the winning config      │
│  python3 -m pytest tests/ -q                             │
│                                                          │
│  # 4. If happy, promote to main config                   │
│  # (copy autoresearch/best_config.py → config/)          │
│  python3 autoresearch/review_results.py --promote        │
└──────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight autoresearch runner")
    parser.add_argument("-n", "--experiments", type=int, default=300,
                        help="Number of experiments (default: 300, ~3hrs)")
    parser.add_argument("--fast", action="store_true",
                        help="Use claude-haiku (faster/cheaper, less accurate proposals)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model explicitly")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous run")
    args = parser.parse_args()

    model = args.model or ("claude-haiku-4-5" if args.fast else "claude-sonnet-4-5")
    run(n_experiments=args.experiments, model=model, resume=args.resume)
