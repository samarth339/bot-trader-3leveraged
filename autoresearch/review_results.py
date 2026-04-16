"""
Review Autoresearch Results
============================
Morning review tool: analyse the results.tsv, show score progression,
compare best config vs baseline, and guide promotion to main.

Usage:
    python3 autoresearch/review_results.py            # full summary
    python3 autoresearch/review_results.py --wins     # wins only
    python3 autoresearch/review_results.py --plot     # ASCII score chart
    python3 autoresearch/review_results.py --diff     # show config diff vs baseline
    python3 autoresearch/review_results.py --promote  # promotion checklist + instructions
    python3 autoresearch/review_results.py --oos      # run best config on OOS period
"""

from __future__ import annotations

import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

_ROOT = Path(__file__).parent.parent
_AR   = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

RESULTS_TSV  = _AR / "results.tsv"
BEST_CONFIG  = _AR / "best_config.py"
MAIN_CONFIG  = _ROOT / "config" / "strategy_config.py"
SNAPSHOTS    = _AR / "snapshots"


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_results() -> List[Dict]:
    if not RESULTS_TSV.exists():
        print("No results.tsv found. Run run_overnight.py first.")
        sys.exit(0)
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def wins(rows: List[Dict]) -> List[Dict]:
    return [r for r in rows if r["result"] == "WIN"]


def best_score(rows: List[Dict]) -> float:
    w = wins(rows)
    return float(w[-1]["score_after"]) if w else float(rows[0].get("score_before", 0))


# ── Display ────────────────────────────────────────────────────────────────────

def _hr(char="─", w=75):
    print(char * w)


def print_summary(rows: List[Dict], wins_only: bool = False) -> None:
    total  = len(rows)
    w_rows = wins(rows)
    f_rows = [r for r in rows if r["result"] == "FAIL"]
    v_rows = [r for r in rows if r["result"] in ("VETO", "SKIP")]
    e_rows = [r for r in rows if r["result"] == "ERROR"]

    _hr("═")
    print("  AUTORESEARCH RESULTS SUMMARY")
    _hr("═")

    # Run stats
    if rows:
        first_ts = rows[0]["timestamp"]
        last_ts  = rows[-1]["timestamp"]
        print(f"  Period     : {first_ts} → {last_ts}")
    print(f"  Total      : {total}  |  Wins: {len(w_rows)}  |  Fails: {len(f_rows)}"
          f"  |  Veto/Skip: {len(v_rows)}  |  Errors: {len(e_rows)}")
    if total > 0:
        print(f"  Win rate   : {len(w_rows)/total*100:.1f}%")

    if rows:
        start_score = float(rows[0]["score_before"])
        end_score   = float(rows[-1]["score_after"]) if rows[-1]["result"] == "WIN" else best_score(rows)
        delta       = end_score - start_score
        print(f"  Score      : {start_score:.5f} → {end_score:.5f}  ({delta:+.5f}  {delta/max(start_score,0.001)*100:+.1f}%)")

    _hr()

    # Experiment table
    display = w_rows if wins_only else rows
    if not display:
        print("  (no results to display)")
        return

    print(f"  {'#':>4}  {'Res':<5}  {'Score':>13}  "
          f"{'Tr-CAGR':>8} {'Tr-DD':>6} {'Val-CAGR':>9} {'Val-DD':>7}  Change")
    _hr()
    for r in display:
        b = float(r["score_before"])
        a = float(r["score_after"])
        d = a - b
        marker = "★" if r["result"] == "WIN" else " "
        print(
            f"  {r['experiment']:>4}  "
            f"{marker}{r['result']:<4}  "
            f"{b:.4f}→{a:.4f}({d:+.4f})  "
            f"{r['train_cagr']:>8} {r['train_dd']:>6}"
            f"{r['val_cagr']:>9} {r['val_dd']:>7}  "
            f"{r['change_desc'][:38]}"
        )

    # Win stack
    if w_rows:
        _hr()
        print("  ACCEPTED CHANGES (stacked):")
        for r in w_rows:
            print(f"    ★ #{r['experiment']:>3}  {r['change_desc']}")
            print(f"           {r['score_before']} → {r['score_after']}  "
                  f"| {r['rationale'][:70]}")


def print_ascii_chart(rows: List[Dict]) -> None:
    """ASCII score progression chart."""
    _hr("═")
    print("  SCORE PROGRESSION")
    _hr("═")

    scores = []
    labels = []
    for r in rows:
        if r["result"] == "WIN":
            scores.append(float(r["score_after"]))
            labels.append(f"#{r['experiment']}")

    if not scores:
        print("  (no wins yet — nothing to chart)")
        return

    # Prepend baseline
    baseline = float(rows[0]["score_before"]) if rows else 0
    scores   = [baseline] + scores
    labels   = ["base"] + labels

    max_s = max(scores)
    min_s = min(scores) - 0.002
    rng   = max(max_s - min_s, 0.001)
    width = 55

    for score, label in zip(scores, labels):
        bar_len = int((score - min_s) / rng * width)
        bar = "█" * bar_len
        print(f"  {label:>5} {score:.4f} |{bar}")

    _hr()


def print_config_diff() -> None:
    """Show what changed between the best config and the original main config."""
    _hr("═")
    print("  CONFIG DIFF  (best vs main)")
    _hr("═")

    if not BEST_CONFIG.exists():
        print("  No best_config.py found (no wins yet).")
        return

    best_lines = BEST_CONFIG.read_text().splitlines()
    main_lines = MAIN_CONFIG.read_text().splitlines()

    best_set = set(best_lines)
    main_set = set(main_lines)

    removed = [l.strip() for l in (main_set - best_set)
               if l.strip() and not l.strip().startswith(("#", '"""', "''"))]
    added   = [l.strip() for l in (best_set - main_set)
               if l.strip() and not l.strip().startswith(("#", '"""', "''"))]

    if not removed and not added:
        print("  No differences found — best config matches main config.")
        return

    for line in removed:
        print(f"  - {line}")
    for line in added:
        print(f"  + {line}")
    _hr()


def run_oos_eval() -> None:
    """Apply best config and run the OOS evaluation."""
    _hr("═")
    print("  OOS EVALUATION — best config on 2022–2025")
    _hr("═")

    if not BEST_CONFIG.exists():
        print("  No best_config.py found. Nothing to evaluate.")
        return

    # Temporarily swap in best config
    import shutil
    backup = MAIN_CONFIG.with_suffix(".py.main_backup")
    shutil.copy2(MAIN_CONFIG, backup)
    shutil.copy2(BEST_CONFIG, MAIN_CONFIG)

    try:
        proc = subprocess.run(
            [sys.executable, str(_AR / "run_backtest.py"), "--json", "--oos"],
            capture_output=True, text=True, timeout=200, cwd=str(_ROOT),
        )
        if proc.returncode == 0:
            result = json.loads(proc.stdout)
            tr = result["train"]
            va = result["val"]
            oo = result.get("oos", {})
            print(f"  Train (2010–2018): CAGR {tr['cagr']*100:.1f}%  DD {tr['max_drawdown']*100:.1f}%  Calmar {tr['calmar']:.3f}")
            print(f"  Val   (2019–2021): CAGR {va['cagr']*100:.1f}%  DD {va['max_drawdown']*100:.1f}%  Calmar {va['calmar']:.3f}")
            print(f"  OOS   (2022–2025): CAGR {oo.get('cagr',0)*100:.1f}%  DD {oo.get('max_drawdown',0)*100:.1f}%  Calmar {oo.get('calmar',0):.3f}")
            print(f"  Composite score  : {result['composite_score']:.5f}")
            verdict = "ACCEPT" if result["accepted"] else "REJECT"
            print(f"  Verdict          : {verdict}")
        else:
            print(f"  Backtest failed: {proc.stderr[-200:]}")
    finally:
        shutil.copy2(backup, MAIN_CONFIG)
        backup.unlink()

    _hr()


def print_promotion_guide(rows: List[Dict]) -> None:
    """Step-by-step guide to promote the best config to main."""
    _hr("═")
    print("  PROMOTION CHECKLIST")
    _hr("═")

    w_rows = wins(rows)
    if not w_rows:
        print("  No wins to promote. Keep running experiments.")
        return

    final_score = float(w_rows[-1]["score_after"])
    baseline    = float(rows[0]["score_before"])
    improvement = (final_score - baseline) / max(baseline, 0.001) * 100

    print(f"""
  Best score   : {final_score:.5f}  (baseline was {baseline:.5f}, +{improvement:.1f}%)
  Best config  : autoresearch/best_config.py
  Snapshots    : autoresearch/snapshots/

  Checklist before promoting:
  ─────────────────────────────────────────────────────────────

  □ 1. Review the config diff
       python3 autoresearch/review_results.py --diff

  □ 2. Run OOS evaluation (should not be awful)
       python3 autoresearch/review_results.py --oos

  □ 3. Run the full test suite on best config
       cp autoresearch/best_config.py config/strategy_config.py
       python3 -m pytest tests/ -q
       (restore if tests fail: cp config/strategy_config.py.bak config/strategy_config.py)

  □ 4. Run robustness tests
       python3 stress_test_robustness.py --no-chart

  □ 5. Update CLAUDE.md with new locked config values

  □ 6. Run shadow mode for 30 days before going live
       python3 shadow_mode.py --backfill 30

  When all boxes are checked:
  ─────────────────────────────────────────────────────────────
  cp autoresearch/best_config.py config/strategy_config.py
  echo "Promoted autoresearch winner (score {final_score:.5f})" >> CLAUDE.md
""")
    _hr()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review autoresearch results")
    parser.add_argument("--wins",    action="store_true", help="Show wins only")
    parser.add_argument("--plot",    action="store_true", help="ASCII score chart")
    parser.add_argument("--diff",    action="store_true", help="Config diff vs main")
    parser.add_argument("--promote", action="store_true", help="Promotion checklist")
    parser.add_argument("--oos",     action="store_true", help="Run OOS evaluation")
    args = parser.parse_args()

    rows = load_results()

    if args.oos:
        run_oos_eval()
    elif args.plot:
        print_ascii_chart(rows)
    elif args.diff:
        print_config_diff()
    elif args.promote:
        print_summary(rows, wins_only=True)
        print_promotion_guide(rows)
    else:
        print_summary(rows, wins_only=args.wins)
        if not args.wins and len(wins(rows)) > 0:
            print()
            print_ascii_chart(rows)
