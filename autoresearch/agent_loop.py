"""
Autoresearch Agent Loop
=======================
Overnight optimization loop: Claude proposes parameter changes, the backtest
evaluates them, winners are kept, losers are reverted.

Works entirely locally — no git required.

Usage:
    python3 autoresearch/agent_loop.py --experiments 100
    python3 autoresearch/agent_loop.py --experiments 600 --model claude-opus-4-6
    python3 autoresearch/agent_loop.py --dry-run --experiments 5   # preview only
    python3 autoresearch/agent_loop.py --resume                    # continue a run

Environment (put in .env at project root):
    ANTHROPIC_API_KEY=sk-ant-...

Throughput: ~30–40 experiments/hour (30s backtest + 5s API + overhead)
Overnight (8 hrs): ~250–320 experiments
"""

from __future__ import annotations   # Python 3.9 compatibility

import sys
import os
import re
import csv
import json
import time
import subprocess
import argparse
import textwrap
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from autoresearch.config_patcher import (
    apply_change, restore_backup, has_backup, diff_summary, read_current_value
)
from autoresearch.safe_ranges import validate, is_dead_end, check_cross_constraints, SAFE_RANGES

# ── Paths ──────────────────────────────────────────────────────────────────────
_AR          = Path(__file__).parent
PROGRAM_MD   = _AR / "program.md"
RESULTS_TSV  = _AR / "results.tsv"
CONFIG_PATH  = _ROOT / "config" / "strategy_config.py"
BACKTEST_SCRIPT = _AR / "run_backtest.py"
RUN_LOG      = _AR / "run.log"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "claude-sonnet-4-5"
MAX_HISTORY        = 20      # lines of results shown in prompt
BACKTEST_TIMEOUT   = 150     # seconds
API_RETRY_DELAYS   = [2, 5, 15]  # exponential backoff on API errors
MIN_IMPROVEMENT    = 0.0005  # must beat baseline by at least this to WIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RUN_LOG), mode="a"),
    ],
)
log = logging.getLogger("autoresearch")


# ── Results Log ───────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "timestamp", "experiment", "change_desc",
    "score_before", "score_after",
    "train_cagr", "train_dd", "train_calmar",
    "val_cagr",   "val_dd",   "val_calmar",
    "result", "rationale",
]

def _init_results() -> None:
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(RESULT_FIELDS)


def _read_results() -> List[Dict]:
    if not RESULTS_TSV.exists():
        return []
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _append_result(
    exp_num: int,
    change_desc: str,
    score_before: float,
    score_after: float,
    metrics: Dict,
    result: str,
    rationale: str,
) -> None:
    train = metrics.get("train", {})
    val   = metrics.get("val", {})
    with open(RESULTS_TSV, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exp_num,
            change_desc,
            f"{score_before:.5f}",
            f"{score_after:.5f}",
            f"{train.get('cagr', 0)*100:.2f}%",
            f"{train.get('max_drawdown', 0)*100:.2f}%",
            f"{train.get('calmar', 0):.3f}",
            f"{val.get('cagr', 0)*100:.2f}%",
            f"{val.get('max_drawdown', 0)*100:.2f}%",
            f"{val.get('calmar', 0):.3f}",
            result,
            rationale[:150].replace("\t", " "),
        ])


# ── Backtest Runner ────────────────────────────────────────────────────────────

def run_backtest() -> Dict:
    """Run run_backtest.py as a subprocess and parse JSON output."""
    try:
        proc = subprocess.run(
            [sys.executable, str(BACKTEST_SCRIPT), "--json"],
            capture_output=True, text=True,
            timeout=BACKTEST_TIMEOUT, cwd=str(_ROOT),
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()[-300:]
            return {"error": stderr or "non-zero exit"}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"error": f"Backtest timed out after {BACKTEST_TIMEOUT}s"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {e}\nstdout: {proc.stdout[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Current Config State ───────────────────────────────────────────────────────

def _read_current_params() -> Dict[str, Any]:
    """Read current values of all tracked parameters from the live config file."""
    params = {}
    for param in SAFE_RANGES:
        ok, val = read_current_value(param)
        if ok:
            params[param] = val
    return params


# ── Agent Prompt Builder ───────────────────────────────────────────────────────

def _format_history(rows: List[Dict]) -> str:
    if not rows:
        return "  (no experiments yet)"
    lines = []
    for r in rows[-MAX_HISTORY:]:
        marker = "★" if r["result"] == "WIN" else "✗" if r["result"] in ("VETO", "ERROR") else "·"
        lines.append(
            f"  {marker} #{r['experiment']:>3}  {r['result']:<5}  "
            f"{r['score_before']}→{r['score_after']}  "
            f"train({r['train_cagr']}/{r['train_dd']})  "
            f"val({r['val_cagr']}/{r['val_dd']})  "
            f"[{r['change_desc'][:55]}]"
        )
    return "\n".join(lines)


def _recent_wins(rows: List[Dict]) -> str:
    wins = [r for r in rows if r["result"] == "WIN"]
    if not wins:
        return "  (no wins yet)"
    return "\n".join(
        f"  ★ Exp #{w['experiment']}: {w['change_desc']} → score {w['score_after']}"
        for w in wins[-5:]
    )


def build_prompt(current_score: float, rows: List[Dict], current_params: Dict) -> str:
    program    = PROGRAM_MD.read_text()
    config     = CONFIG_PATH.read_text()
    history    = _format_history(rows)
    wins       = _recent_wins(rows)

    # Already-tried changes (to avoid repeats)
    tried = set()
    for r in rows:
        tried.add(r["change_desc"])
    tried_str = "\n".join(f"  - {t}" for t in list(tried)[-30:]) if tried else "  (none)"

    return textwrap.dedent(f"""
        You are an expert algorithmic trading strategy optimizer running inside
        an autoresearch loop. Your goal is to improve a TQQQ/SQQQ dual-portfolio
        strategy by proposing one precise parameter change at a time.

        ## Research Program (constraints + baseline)
        {program}

        ## Current Strategy Config (the file you can modify)
        ```python
        {config}
        ```

        ## Current Best Score
        {current_score:.5f}  (composite = sqrt(train_Calmar × val_Calmar))

        ## Experiment History (last {MAX_HISTORY})
        {history}

        ## Accepted Wins So Far
        {wins}

        ## Already-Tried Changes (DO NOT REPEAT THESE)
        {tried_str}

        ## Your Task
        Propose exactly ONE parameter change that has NOT been tried yet and
        is likely to improve the composite score based on the history above.

        Think step by step:
        1. Which parameter has the most room to improve? Look at train vs val gaps.
        2. Is the bottleneck CAGR, MaxDD, or Calmar? Each suggests different levers.
        3. What does the history say — which directions improve and which hurt?
        4. Stay within safe ranges. Never repeat a tried change.

        Output ONLY a JSON object — no explanation outside the JSON:

        {{
          "param": "STRATEGY_A_CONFIG.vix_exit",
          "new_value": 23,
          "change_desc": "Strategy A vix_exit: 25 → 23",
          "rationale": "Val period has high MaxDD (55.5%). Lowering A's exit threshold reduces exposure earlier during VIX spikes. Train period was mostly calm so this should preserve train performance."
        }}

        Valid param names:
          REGIME_CONFIG.{{ma_window, vix_bull, vix_hi_vol, vix_smooth}}
          ALLOC_CONFIG.{{bull, uncertain, high_vol}}  — value must be [A_weight, B_weight]
          STRATEGY_A_CONFIG.{{ma_long, vix_exit, vix_reentry, confirm_bars, max_position_pct}}
          STRATEGY_B_CONFIG.{{ma_long, vix_exit, vix_reentry, confirm_bars, max_position_pct, crash_brake_pct}}

        NEVER modify: t1_execution, stagger_exit, vol_scale, crash_brake_pct for Strategy A
    """).strip()


# ── Claude API ─────────────────────────────────────────────────────────────────

def ask_claude(prompt: str, model: str) -> Optional[Dict]:
    """Call Claude API. Returns parsed JSON proposal or None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — cannot call Claude API")
        return None
    if not _ANTHROPIC_AVAILABLE:
        log.error("anthropic package not installed — run: pip3 install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    for attempt, delay in enumerate([0] + API_RETRY_DELAYS):
        if delay:
            log.info(f"  API retry {attempt} in {delay}s...")
            time.sleep(delay)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            # Extract first {...} block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error (attempt {attempt+1}): {e}")
            log.debug(f"  Raw response: {raw[:300]}")
        except anthropic.RateLimitError:
            log.warning("  Rate limit hit, backing off...")
        except anthropic.APIError as e:
            log.warning(f"  API error: {e}")
        except Exception as e:
            log.warning(f"  Unexpected error: {e}")
            break

    return None


# ── Main Loop ──────────────────────────────────────────────────────────────────

def run_loop(
    max_experiments: int = 100,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    resume: bool = False,
) -> None:

    _init_results()

    log.info("=" * 65)
    log.info("  AUTORESEARCH AGENT LOOP")
    log.info(f"  Model      : {model}")
    log.info(f"  Experiments: {max_experiments}")
    log.info(f"  Mode       : {'DRY RUN' if dry_run else 'LIVE'}")
    log.info(f"  Log file   : {RUN_LOG}")
    log.info("=" * 65)

    # ── Establish baseline ─────────────────────────────────────────────────────
    existing = _read_results()
    if resume and existing:
        wins = [r for r in existing if r["result"] == "WIN"]
        current_score = float(wins[-1]["score_after"]) if wins else float(existing[0]["score_before"])
        exp_offset = int(existing[-1]["experiment"])
        log.info(f"[resume] {len(existing)} experiments already done, score={current_score:.5f}")
    else:
        log.info("[init] Running baseline backtest...")
        baseline = run_backtest()
        if "error" in baseline:
            log.error(f"Baseline failed: {baseline['error']}")
            sys.exit(1)
        current_score = baseline["composite_score"]
        exp_offset = 0
        log.info(f"  Baseline score : {current_score:.5f}")
        log.info(
            f"  Train: CAGR {baseline['train']['cagr']*100:.1f}%  "
            f"MaxDD {baseline['train']['max_drawdown']*100:.1f}%  "
            f"Calmar {baseline['train']['calmar']:.3f}"
        )
        log.info(
            f"  Val  : CAGR {baseline['val']['cagr']*100:.1f}%  "
            f"MaxDD {baseline['val']['max_drawdown']*100:.1f}%  "
            f"Calmar {baseline['val']['calmar']:.3f}"
        )

    wins_total = fails_total = vetos_total = errors_total = 0

    for i in range(1, max_experiments + 1):
        exp_num = exp_offset + i
        log.info(f"\n[{i:>3}/{max_experiments}] exp#{exp_num}  score={current_score:.5f}  "
                 f"W={wins_total} F={fails_total} V={vetos_total}")

        rows = _read_results()
        current_params = _read_current_params()

        # ── Ask Claude ─────────────────────────────────────────────────────────
        prompt   = build_prompt(current_score, rows, current_params)
        proposal = ask_claude(prompt, model)

        if proposal is None:
            log.warning("  No valid proposal from Claude, skipping experiment")
            time.sleep(5)
            continue

        param       = proposal.get("param", "").strip()
        new_value   = proposal.get("new_value")
        change_desc = proposal.get("change_desc", f"{param} → {new_value}").strip()
        rationale   = proposal.get("rationale", "").strip()

        log.info(f"  Proposed : {change_desc}")
        log.info(f"  Rationale: {rationale[:100]}")

        # ── Pre-flight checks ──────────────────────────────────────────────────
        # 1. Dead-end check
        dead, dead_reason = is_dead_end(param, new_value)
        if dead:
            log.info(f"  SKIP (dead end): {dead_reason}")
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "SKIP", dead_reason)
            continue

        # 2. Safe range check
        ok, reason = validate(param, new_value)
        if not ok:
            log.info(f"  SKIP (out of range): {reason}")
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "SKIP", reason)
            continue

        # 3. Cross-constraint check (simulate the updated param)
        sim_params = dict(current_params)
        sim_params[param] = new_value
        cross_ok, cross_reason = check_cross_constraints(sim_params)
        if not cross_ok:
            log.info(f"  SKIP (cross-constraint): {cross_reason}")
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "SKIP", cross_reason)
            continue

        # 4. Duplicate check
        already_tried = any(r["change_desc"] == change_desc for r in rows)
        if already_tried:
            log.info(f"  SKIP (already tried): {change_desc}")
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "SKIP", "duplicate")
            continue

        if dry_run:
            log.info("  [dry-run] Not applying.")
            continue

        # ── Apply change ───────────────────────────────────────────────────────
        patch_ok, patch_reason = apply_change(param, new_value)
        if not patch_ok:
            log.warning(f"  PATCH FAILED: {patch_reason}")
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "ERROR", f"patch: {patch_reason}")
            errors_total += 1
            continue

        actual_diff = diff_summary()
        log.info(f"  Applied  : {actual_diff}")

        # ── Evaluate ───────────────────────────────────────────────────────────
        t0 = time.time()
        eval_result = run_backtest()
        elapsed = time.time() - t0
        log.info(f"  Backtest : {elapsed:.0f}s")

        if "error" in eval_result:
            log.warning(f"  BACKTEST ERROR: {eval_result['error'][:120]}")
            restore_backup()
            _append_result(exp_num, change_desc, current_score, 0.0,
                           {}, "ERROR", eval_result["error"][:120])
            errors_total += 1
            continue

        new_score = eval_result["composite_score"]
        accepted  = eval_result["accepted"]
        viols     = eval_result.get("veto_violations", [])

        t_cagr = eval_result["train"]["cagr"] * 100
        t_dd   = eval_result["train"]["max_drawdown"] * 100
        t_cal  = eval_result["train"]["calmar"]
        v_cagr = eval_result["val"]["cagr"] * 100
        v_dd   = eval_result["val"]["max_drawdown"] * 100
        v_cal  = eval_result["val"]["calmar"]

        log.info(
            f"  Score    : {current_score:.5f} → {new_score:.5f}  "
            f"(Δ{new_score - current_score:+.5f})"
        )
        log.info(
            f"  Train    : CAGR {t_cagr:.1f}%  DD {t_dd:.1f}%  Cal {t_cal:.3f}"
        )
        log.info(
            f"  Val      : CAGR {v_cagr:.1f}%  DD {v_dd:.1f}%  Cal {v_cal:.3f}"
        )

        # ── Accept or reject ───────────────────────────────────────────────────
        if viols:
            log.info(f"  VETO: {' | '.join(viols)}")
            restore_backup()
            vetos_total += 1
            _append_result(exp_num, change_desc, current_score, new_score,
                           eval_result, "VETO", " | ".join(viols)[:150])

        elif new_score > current_score + MIN_IMPROVEMENT:
            log.info(f"  ★ WIN  → keeping change, new best = {new_score:.5f}")
            current_score = new_score
            wins_total += 1
            _append_result(exp_num, change_desc, current_score, new_score,
                           eval_result, "WIN", rationale[:150])
            # Save snapshot of winning config
            _save_best_snapshot(exp_num, new_score)

        else:
            restore_backup()
            fails_total += 1
            _append_result(exp_num, change_desc, current_score, new_score,
                           eval_result, "FAIL", rationale[:150])

        time.sleep(1)   # brief pause between experiments

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  AUTORESEARCH COMPLETE")
    log.info(f"  Experiments  : {max_experiments}")
    log.info(f"  Wins         : {wins_total}")
    log.info(f"  Fails        : {fails_total}")
    log.info(f"  Vetos/Skips  : {vetos_total}")
    log.info(f"  Errors       : {errors_total}")
    log.info(f"  Final score  : {current_score:.5f}")
    log.info(f"  Results log  : {RESULTS_TSV}")
    log.info(f"  Best config  : {_AR / 'best_config.py'}")
    log.info("=" * 65)

    _notify_completion(wins_total, fails_total, current_score, max_experiments)


def _save_best_snapshot(exp_num: int, score: float) -> None:
    """Snapshot the current winning config for easy review."""
    snap_dir = _AR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    dest = snap_dir / f"exp{exp_num:04d}_score{score:.4f}_strategy_config.py"
    import shutil
    shutil.copy2(CONFIG_PATH, dest)
    # Always keep a "best_config.py" pointing to current best
    shutil.copy2(CONFIG_PATH, _AR / "best_config.py")
    log.info(f"  Snapshot : {dest.name}")


def _notify_completion(wins: int, fails: int, final_score: float, n_exp: int) -> None:
    """Send email summary if send_email.py is configured."""
    try:
        sys.path.insert(0, str(_ROOT))
        from send_email import send_email
        body = (
            f"Autoresearch run complete.\n\n"
            f"Experiments : {n_exp}\n"
            f"Wins        : {wins}\n"
            f"Fails       : {fails}\n"
            f"Final score : {final_score:.5f}\n\n"
            f"Review results:\n"
            f"  python3 autoresearch/review_results.py --wins\n\n"
            f"Winning configs are in:\n"
            f"  autoresearch/snapshots/\n"
            f"  autoresearch/best_config.py\n"
        )
        send_email("Autoresearch Complete", body)
        log.info("  Email summary sent.")
    except Exception as e:
        log.debug(f"Email not sent: {e}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoresearch agent loop — overnight strategy parameter optimizer"
    )
    parser.add_argument("--experiments", "-n", type=int, default=100,
                        help="Number of experiments (default: 100)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Propose changes but don't apply or backtest them")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous run, continuing experiment count")
    args = parser.parse_args()

    try:
        run_loop(
            max_experiments=args.experiments,
            model=args.model,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        log.info("\n[interrupted] Restoring last committed config...")
        if has_backup():
            restore_backup()
            log.info("  Config restored to last accepted state.")
        log.info(f"  Results saved to: {RESULTS_TSV}")
        log.info("  Resume with: python3 autoresearch/agent_loop.py --resume")
