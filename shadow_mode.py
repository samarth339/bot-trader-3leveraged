"""
Shadow Mode Runner — 30-Day Automated Testing
==============================================
Runs daily with ZERO human intervention:
  1. Refreshes market data (yfinance)
  2. Computes today's signal (T-1, no look-ahead)
  3. Detects regime changes and anomalies
  4. Sends email alert on regime flip or danger signal
  5. Sends weekly summary every Friday
  6. Generates final 30-day report on day 30

Email alerts go to: samarth339@gmail.com

Usage:
    python shadow_mode.py              # normal daily run
    python shadow_mode.py --report     # force-generate the 30-day report
    python shadow_mode.py --status     # print current shadow-mode state
    python shadow_mode.py --reset      # wipe state and start fresh (day 0)

Designed to be called by a cron/scheduled task — exits cleanly with:
    0 = success
    1 = data error
    2 = anomaly detected (still logs and emails)
"""

import sys
import json
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Email (Gmail MCP is called via subprocess to avoid import complexity) ───
ALERT_EMAIL = "samarth339@gmail.com"

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
LOGS_DIR    = ROOT / "logs"
STATE_FILE  = LOGS_DIR / "shadow_state.json"
HISTORY_CSV = LOGS_DIR / "signal_history.csv"
REPORT_DIR  = LOGS_DIR / "reports"

LOGS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "shadow_mode.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("shadow_mode")

# ── Thresholds ────────────────────────────────────────────────────────────────
VIX_DANGER          = 35.0   # send alert if VIX exceeds this
VIX_EXTREME         = 45.0   # send CRITICAL alert
SHADOW_DAYS         = 30     # total shadow-mode period
FRIDAY_WEEKDAY      = 4      # 0=Mon … 4=Fri


# ──────────────────────────────────────────────────────────────────────────────
# State management
# ──────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "start_date":    None,
        "day_number":    0,
        "last_run_date": None,
        "prev_regime":   None,
        "total_alerts":  0,
        "regime_flips":  0,
        "days_bull":     0,
        "days_uncertain":0,
        "days_high_vol": 0,
        "completed":     False,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Data refresh
# ──────────────────────────────────────────────────────────────────────────────

def refresh_data() -> bool:
    """
    Download latest market data via fetch_data.py.
    Returns True on success.
    """
    log.info("Refreshing market data...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "data" / "fetch_data.py")],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log.info("Data refresh complete.")
            return True
        else:
            log.error(f"Data refresh failed:\n{result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        log.error("Data refresh timed out (>120s)")
        return False
    except FileNotFoundError:
        log.warning("fetch_data.py not found — using cached data.")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Signal computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_today_signal() -> dict:
    """Run daily_signal.py --shadow and parse the result from signal_history.csv."""
    from daily_signal import load_data, compute_regime, resolve_action, append_signal_log
    from config.strategy_config import RISK_CONFIG

    today = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    qqq, vix = load_data()
    sig = compute_regime(qqq, vix, today)

    # Load previous regime and pct_vs_sma for dynamic uncertain allocation
    prev_regime     = sig["regime"]  # default: no change
    prev_pct_vs_sma = 0.0
    if HISTORY_CSV.exists():
        try:
            hist = pd.read_csv(HISTORY_CSV)
            if not hist.empty:
                last = hist.iloc[-1]
                prev_regime = last.get("regime", sig["regime"])
                raw_pct = last.get("pct_vs_sma", 0.0)
                prev_pct_vs_sma = float(raw_pct) if raw_pct not in ("", None) else 0.0
        except Exception:
            pass

    cur_pct_vs_sma = sig["pct_vs_sma"] if not np.isnan(sig["pct_vs_sma"]) else 0.0
    action = resolve_action(sig["regime"], prev_regime,
                            pct_vs_sma=cur_pct_vs_sma,
                            prev_pct_vs_sma=prev_pct_vs_sma)

    row = {
        "as_of_date":   sig["as_of_date"].strftime("%Y-%m-%d"),
        "signal_date":  sig["signal_date"].strftime("%Y-%m-%d"),
        "regime":       sig["regime"],
        "action":       action["action"],
        "weight_a":     action["target_alloc"][0],
        "weight_b":     action["target_alloc"][1],
        "rebalance":    action["rebalance_needed"],
        "drift_pct":    action["drift_pct"],
        "qqq_price":    round(sig["price_signal"], 4),
        "sma_val":      round(sig["sma_val"], 4) if not np.isnan(sig["sma_val"]) else "",
        "pct_vs_sma":   round(sig["pct_vs_sma"], 2) if not np.isnan(sig["pct_vs_sma"]) else "",
        "vix_signal":   round(sig["vix_signal"], 2),
        "vix_raw":      round(sig["vix_raw"], 2) if not np.isnan(sig["vix_raw"]) else "",
        "shadow":       True,
    }
    append_signal_log(row)
    return {**sig, **row, "action_dict": action, "prev_regime": prev_regime}


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_anomalies(signal: dict, state: dict) -> list:
    """
    Returns list of anomaly dicts: {level, code, message}
    Levels: INFO, WARNING, CRITICAL
    """
    alerts = []
    regime    = signal["regime"]
    prev      = signal.get("prev_regime", regime)
    vix_raw   = signal.get("vix_raw", 0) or 0
    vix_sig   = signal.get("vix_signal", 0) or 0

    # Regime flip
    if regime != prev and prev is not None:
        alerts.append({
            "level":   "WARNING",
            "code":    "REGIME_FLIP",
            "message": f"Regime changed: {prev.upper()} → {regime.upper()}  "
                       f"(VIX={vix_raw:.1f}, action={signal.get('action','?')})",
        })

    # VIX danger
    if vix_raw >= VIX_EXTREME:
        alerts.append({
            "level":   "CRITICAL",
            "code":    "VIX_EXTREME",
            "message": f"EXTREME VIX: {vix_raw:.1f} ≥ {VIX_EXTREME} — market in panic",
        })
    elif vix_raw >= VIX_DANGER:
        alerts.append({
            "level":   "WARNING",
            "code":    "VIX_DANGER",
            "message": f"High VIX: {vix_raw:.1f} ≥ {VIX_DANGER} — elevated fear",
        })

    # Rapid regime flip (flip back within 2 days)
    if state["regime_flips"] > 0 and regime != prev:
        alerts.append({
            "level":   "WARNING",
            "code":    "WHIPSAW",
            "message": f"Potential whipsaw: {state['regime_flips']} flips in {state['day_number']} days",
        })

    return alerts


# ──────────────────────────────────────────────────────────────────────────────
# Email helpers
# ──────────────────────────────────────────────────────────────────────────────

def _regime_emoji(regime: str) -> str:
    return {"bull": "🐂", "uncertain": "⚠️", "high_vol": "🔴"}.get(regime, "❓")


def _alloc_str(wa: float, wb: float) -> str:
    return f"{int(wa*100)}% A (aggressive) / {int(wb*100)}% B (defensive)"


def send_alert_email(signal: dict, anomalies: list, day_number: int):
    """Send an alert email via Gmail SMTP for anomalies/regime flips."""
    from send_email import send_email as _send

    date_str = signal.get("as_of_date", datetime.today().strftime("%Y-%m-%d"))
    regime   = signal.get("regime", "unknown")
    emoji    = _regime_emoji(regime)
    levels   = [a["level"] for a in anomalies]
    severity = "CRITICAL" if "CRITICAL" in levels else "WARNING"

    subject = (f"[TRADING BOT] {severity} — {emoji} {regime.upper()} "
               f"— Day {day_number} — {date_str}")

    body_lines = [
        f"Trading Bot Shadow Mode — Day {day_number}/{SHADOW_DAYS}",
        f"Date: {date_str}",
        "",
        "── SIGNAL ──────────────────────────────────────────────",
        f"Regime:        {_regime_emoji(regime)} {regime.upper()}",
        f"Prev Regime:   {signal.get('prev_regime','?').upper()}",
        f"Allocation:    {_alloc_str(signal.get('weight_a',0.5), signal.get('weight_b',0.5))}",
        f"Action:        {signal.get('action','?')}",
        f"QQQ (T-1):     ${signal.get('qqq_price', 0):.2f}  "
        f"(SMA-150: ${signal.get('sma_150', 0):.2f}  {signal.get('pct_vs_sma', 0):+.1f}%)",
        f"VIX (smoothed):{signal.get('vix_signal', 0):.1f}  (raw: {signal.get('vix_raw', 0):.1f})",
        "",
        "── ALERTS ──────────────────────────────────────────────",
    ]
    for a in anomalies:
        body_lines.append(f"  [{a['level']}] {a['code']}: {a['message']}")

    body_lines += [
        "",
        "── SHADOW MODE ─────────────────────────────────────────",
        "No trades have been placed. This is observation only.",
        f"Log: {HISTORY_CSV}",
        "",
        "To view full history:  cat logs/signal_history.csv",
        "To run report now:     python shadow_mode.py --report",
    ]

    body = "\n".join(body_lines)

    # Send immediately; also write pending file as fallback record
    ok = _send(subject, body, ALERT_EMAIL)
    status = "sent" if ok else "FAILED — check logs/send_email errors"
    log.info(f"Email alert {status}: {subject}")
    pending = LOGS_DIR / "pending_email.json"
    with open(pending, "w") as f:
        json.dump({"to": ALERT_EMAIL, "subject": subject, "body": body, "sent": ok}, f, indent=2)
    return subject, body


def compute_shadow_pnl(state: dict) -> dict:
    """
    Run DualPortfolioBacktester over the full history, then normalize the equity
    curve to $5,000 at the shadow start date and return P&L stats for the shadow
    window only.  Returns {} silently on any error — non-fatal.
    """
    try:
        from backtester.dual_portfolio import DualPortfolioBacktester
        from strategies.long_only_guard_v2 import LongOnlyGuardV2
        from config.settings import INITIAL_CAPITAL
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS

        DATA_DIR = ROOT / "data" / "processed"
        tqqq = pd.read_csv(DATA_DIR / "TQQQ_full.csv", index_col=0, parse_dates=True)["Close"]
        sqqq = pd.read_csv(DATA_DIR / "SQQQ_full.csv", index_col=0, parse_dates=True)["Close"]
        qqq  = pd.read_csv(DATA_DIR / "QQQ_full.csv",  index_col=0, parse_dates=True)["Close"]
        vix  = pd.read_csv(DATA_DIR / "VIX_full.csv",  index_col=0, parse_dates=True)["Close"]

        sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
        sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})

        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=INITIAL_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )
        result  = dp.run()
        ec_full = result["equity_curve"]["equity"]

        shadow_start = pd.Timestamp(state.get("start_date", datetime.today().strftime("%Y-%m-%d")))
        shadow_end   = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))

        # Value just before shadow period started → use as normalization base
        pre = ec_full[ec_full.index < shadow_start]
        if pre.empty:
            return {}
        base_val = float(pre.iloc[-1])

        SEED      = 5_000.0
        shadow_ec = ec_full[ec_full.index >= shadow_start] / base_val * SEED
        shadow_ec = shadow_ec[shadow_ec.index <= shadow_end]
        if len(shadow_ec) < 2:
            return {}

        daily_ret   = shadow_ec.pct_change().dropna()
        current_val = float(shadow_ec.iloc[-1])
        prev_val    = float(shadow_ec.iloc[-2])

        return {
            "start_value":      SEED,
            "current_value":    round(current_val, 2),
            "gain_loss":        round(current_val - SEED, 2),
            "pct_return":       round((current_val - SEED) / SEED * 100, 2),
            "today_change_val": round(current_val - prev_val, 2),
            "today_change_pct": round((current_val - prev_val) / prev_val * 100, 2),
            "best_day_pct":     round(float(daily_ret.max()) * 100, 2),
            "worst_day_pct":    round(float(daily_ret.min()) * 100, 2),
        }
    except Exception as e:
        log.warning(f"Shadow P&L computation skipped: {e}")
        return {}


def send_daily_digest(signal: dict, state: dict, day_number: int, anomalies: list,
                      pnl: dict = None):
    """Send a brief daily digest email every single run — no conditions."""
    from send_email import send_email as _send

    date_str = signal.get("as_of_date", datetime.today().strftime("%Y-%m-%d"))
    regime   = signal.get("regime", "unknown")
    emoji    = _regime_emoji(regime)
    vix_raw  = signal.get("vix_raw",  0) or 0
    vix_sig  = signal.get("vix_signal", 0) or 0
    qqq      = signal.get("qqq_price", 0) or 0
    sma      = signal.get("sma_val",   0) or 0
    pct_sma  = signal.get("pct_vs_sma", 0) or 0
    wa       = signal.get("weight_a",  0.5)
    wb       = signal.get("weight_b",  0.5)
    action   = signal.get("action", "?")

    total   = max(state["day_number"], 1)
    pct_b   = round(state["days_bull"]      / total * 100, 1)
    pct_u   = round(state["days_uncertain"] / total * 100, 1)
    pct_h   = round(state["days_high_vol"]  / total * 100, 1)

    anomaly_block = ""
    if anomalies:
        anomaly_block = "\n── ALERTS ──────────────────────────────────────────────\n"
        for a in anomalies:
            anomaly_block += f"  [{a['level']}] {a['code']}: {a['message']}\n"

    # ── Portfolio snapshot block ──────────────────────────────────────────────
    if pnl:
        gain       = pnl["gain_loss"]
        gain_sign  = "+" if gain >= 0 else ""
        td_sign    = "+" if pnl["today_change_val"] >= 0 else ""
        pnl_block  = f"""
── PORTFOLIO SNAPSHOT (shadow / paper) ─────────────────────
  Starting Capital     ${pnl['start_value']:>9,.2f}
  Current Value        ${pnl['current_value']:>9,.2f}
  Total Return         {gain_sign}${abs(gain):>8,.2f}   ({gain_sign}{pnl['pct_return']:.1f}%)

  Today's Change       {td_sign}${abs(pnl['today_change_val']):>8,.2f}   ({td_sign}{pnl['today_change_pct']:.1f}%)
  Best Single Day      +{pnl['best_day_pct']:.1f}%
  Worst Single Day     {pnl['worst_day_pct']:.1f}%
─────────────────────────────────────────────────────────────
  No real money — shadow observation only"""
    else:
        pnl_block = "\n  (Portfolio P&L unavailable — data not fetched yet)"

    subject = (f"[Bot Day {day_number}] {emoji} {regime.upper()} "
               f"| VIX {vix_raw:.1f} | QQQ ${qqq:.2f} | {action} — {date_str}")

    body = f"""Trading Bot — Daily Report
Day {day_number}/{SHADOW_DAYS}  |  {date_str}
{pnl_block}

── TODAY'S SIGNAL ───────────────────────────────────────────
  Regime:        {emoji} {regime.upper()}
  Action:        {action}
  Allocation:    {_alloc_str(wa, wb)}
  QQQ (T-1):     ${qqq:.2f}  (SMA-130: ${sma:.2f}  {pct_sma:+.1f}% vs SMA)
  VIX smoothed:  {vix_sig:.1f}   (raw: {vix_raw:.1f})
{anomaly_block}
── SHADOW STATS (since Mar 27) ──────────────────────────────
  BULL:      {pct_b:.1f}%  ({state['days_bull']} days)
  UNCERTAIN: {pct_u:.1f}%  ({state['days_uncertain']} days)
  HIGH-VOL:  {pct_h:.1f}%  ({state['days_high_vol']} days)
  Regime flips:  {state['regime_flips']}
  Days left:     {SHADOW_DAYS - day_number}

Full log: https://github.com/samarth339/bot-trader-3leveraged/commits/main
"""

    ok = _send(subject, body, ALERT_EMAIL)
    log.info(f"Daily digest {'sent' if ok else 'FAILED'}: {subject}")


def send_weekly_summary(state: dict, day_number: int):
    """Build and send weekly summary email."""
    from send_email import send_email as _send

    date_str = datetime.today().strftime("%Y-%m-%d")
    subject  = f"[TRADING BOT] Weekly Summary — Day {day_number} — {date_str}"

    total = state["day_number"]
    pct_bull = round(state["days_bull"]     / max(total, 1) * 100, 1)
    pct_unc  = round(state["days_uncertain"]/ max(total, 1) * 100, 1)
    pct_hv   = round(state["days_high_vol"] / max(total, 1) * 100, 1)

    # Load recent history
    history_lines = []
    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV).tail(7)
        history_lines = ["── LAST 7 DAYS ──────────────────────────────────────"]
        for _, row in hist.iterrows():
            emoji = _regime_emoji(str(row.get("regime", "")))
            history_lines.append(
                f"  {row.get('as_of_date','?')}  {emoji} {str(row.get('regime','?')).upper():<10}"
                f"  VIX={row.get('vix_raw','?'):<6}  Action={row.get('action','?')}"
            )

    body_lines = [
        f"Trading Bot — Weekly Shadow Mode Summary",
        f"Week ending: {date_str}  (Day {day_number}/{SHADOW_DAYS})",
        "",
        "── REGIME DISTRIBUTION (since start) ───────────────────",
        f"  BULL:     {pct_bull:>5.1f}%  ({state['days_bull']} days)",
        f"  UNCERTAIN:{pct_unc:>5.1f}%  ({state['days_uncertain']} days)",
        f"  HIGH-VOL: {pct_hv:>5.1f}%  ({state['days_high_vol']} days)",
        f"  Regime flips total: {state['regime_flips']}",
        f"  Alerts sent:        {state['total_alerts']}",
        "",
    ] + history_lines + [
        "",
        "── STATUS ───────────────────────────────────────────────",
        f"  Shadow mode is {'ACTIVE' if not state['completed'] else 'COMPLETE'}.",
        f"  {SHADOW_DAYS - day_number} days remaining.",
        "",
        "To view full log:   cat logs/shadow_mode.log",
        "To view signals:    cat logs/signal_history.csv",
    ]

    body = "\n".join(body_lines)
    ok = _send(subject, body, ALERT_EMAIL)
    status = "sent" if ok else "FAILED — check logs/send_email errors"
    log.info(f"Weekly summary {status}: {subject}")
    pending = LOGS_DIR / "pending_weekly_email.json"
    with open(pending, "w") as f:
        json.dump({"to": ALERT_EMAIL, "subject": subject, "body": body, "sent": ok}, f, indent=2)
    return subject, body


# ──────────────────────────────────────────────────────────────────────────────
# 30-day report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(state: dict) -> str:
    """Generate the final 30-day shadow mode report and return path."""
    date_str    = datetime.today().strftime("%Y%m%d")
    report_path = REPORT_DIR / f"shadow_report_{date_str}.txt"

    total = state["day_number"]
    pct_b = round(state["days_bull"]      / max(total, 1) * 100, 1)
    pct_u = round(state["days_uncertain"] / max(total, 1) * 100, 1)
    pct_h = round(state["days_high_vol"]  / max(total, 1) * 100, 1)

    lines = [
        "=" * 60,
        f"  TRADING BOT — 30-DAY SHADOW MODE REPORT",
        f"  Start: {state['start_date']}  |  End: {datetime.today().strftime('%Y-%m-%d')}",
        "=" * 60,
        "",
        "REGIME DISTRIBUTION",
        f"  BULL:      {pct_b:>5.1f}%  ({state['days_bull']} days)",
        f"  UNCERTAIN: {pct_u:>5.1f}%  ({state['days_uncertain']} days)",
        f"  HIGH-VOL:  {pct_h:>5.1f}%  ({state['days_high_vol']} days)",
        "",
        "SYSTEM BEHAVIOUR",
        f"  Regime flips:    {state['regime_flips']}",
        f"  Alerts fired:    {state['total_alerts']}",
        f"  Days completed:  {total}",
        "",
        "SIGNAL HISTORY",
    ]

    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV)
        lines.append(f"  Total signals logged: {len(hist)}")
        lines.append("")
        lines.append(f"  {'Date':<12} {'Regime':<10} {'Action':<14} {'VIX':>6} {'QQQ':>8} {'Alloc':>10}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*6} {'-'*8} {'-'*10}")
        for _, row in hist.iterrows():
            wa = row.get("weight_a", 0.5)
            alloc_str = f"{int(float(wa)*100)}/{int((1-float(wa))*100)}"
            lines.append(
                f"  {str(row.get('as_of_date','')):<12} "
                f"{str(row.get('regime','')).upper():<10} "
                f"{str(row.get('action','')):<14} "
                f"{str(row.get('vix_raw',''))[:6]:>6} "
                f"${str(row.get('qqq_price',''))[:7]:>7} "
                f"{alloc_str:>10}"
            )

    lines += [
        "",
        "ASSESSMENT",
        _assess(state, pct_b, pct_u, pct_h),
        "",
        "NEXT STEPS",
        "  If system looks stable → proceed to paper trading.",
        "  If excessive regime flips → review VIX thresholds.",
        "  If HIGH-VOL > 40% → market conditions may be unfavourable.",
        "=" * 60,
    ]

    report_path.write_text("\n".join(lines))
    log.info(f"30-day report written: {report_path}")
    return str(report_path)


def _assess(state, pct_b, pct_u, pct_h) -> str:
    issues = []
    if state["regime_flips"] > 15:
        issues.append(f"  ⚠ High flip count ({state['regime_flips']}) — possible whipsaw sensitivity")
    if pct_h > 40:
        issues.append(f"  ⚠ HIGH-VOL dominated ({pct_h}%) — unfavourable period for this strategy")
    if state["total_alerts"] > 10:
        issues.append(f"  ⚠ Many alerts ({state['total_alerts']}) — review email fatigue")
    if not issues:
        return "  ✅ System behaved as expected. Ready for paper trading."
    return "\n".join(issues)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Force generate 30-day report")
    parser.add_argument("--status", action="store_true", help="Print current shadow state")
    parser.add_argument("--reset",  action="store_true", help="Reset shadow state (day 0)")
    parser.add_argument("--force",  action="store_true", help="Re-run even if already ran today (for testing)")
    args = parser.parse_args()

    state = load_state()

    # ── Status ────────────────────────────────────────────────────────────────
    if args.status:
        print(json.dumps(state, indent=2, default=str))
        return 0

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        STATE_FILE.unlink(missing_ok=True)
        HISTORY_CSV.unlink(missing_ok=True)
        log.info("Shadow state reset. Run without --reset to start day 1.")
        return 0

    # ── Force report ──────────────────────────────────────────────────────────
    if args.report:
        path = generate_report(state)
        print(f"Report: {path}")
        return 0

    # ── Normal daily run ──────────────────────────────────────────────────────
    today_str = datetime.today().strftime("%Y-%m-%d")

    # Skip if already ran today (bypass with --force)
    if state["last_run_date"] == today_str and not args.force:
        log.info(f"Already ran today ({today_str}). Skipping. Use --force to override.")
        return 0

    # Initialise start date
    if state["start_date"] is None:
        state["start_date"] = today_str
        log.info(f"Shadow mode started — Day 1 of {SHADOW_DAYS}")

    state["day_number"]    += 1
    state["last_run_date"]  = today_str
    day_number = state["day_number"]

    log.info(f"{'='*50}")
    log.info(f"Shadow Mode Day {day_number}/{SHADOW_DAYS}  —  {today_str}")

    # Refresh data
    data_ok = refresh_data()
    if not data_ok:
        log.error("Data refresh failed — aborting today's run.")
        save_state(state)
        return 1

    # Compute signal
    try:
        signal = compute_today_signal()
    except Exception as e:
        log.error(f"Signal computation failed: {e}")
        save_state(state)
        return 1

    regime = signal["regime"]
    log.info(f"Regime: {regime.upper()}  |  Action: {signal.get('action','?')}  "
             f"|  VIX: {signal.get('vix_raw', 0):.1f}")

    # Update state counters
    prev_regime = state["prev_regime"]
    if regime != prev_regime and prev_regime is not None:
        state["regime_flips"] += 1
        log.info(f"REGIME FLIP: {prev_regime.upper()} → {regime.upper()}")
    state["prev_regime"] = regime
    state[f"days_{regime}"] = state.get(f"days_{regime}", 0) + 1

    # Detect anomalies
    anomalies = detect_anomalies(signal, state)
    for a in anomalies:
        log.log(
            logging.CRITICAL if a["level"] == "CRITICAL" else logging.WARNING,
            f"[{a['code']}] {a['message']}"
        )

    # Send alert email if any anomalies
    if anomalies:
        state["total_alerts"] += 1
        send_alert_email(signal, anomalies, day_number)

    # Compute shadow P&L (backtester over shadow window, normalized to $5K)
    pnl = compute_shadow_pnl(state)
    if pnl:
        state["portfolio"] = pnl
        log.info(f"Shadow P&L: ${pnl['current_value']:,.2f}  "
                 f"({'+' if pnl['gain_loss']>=0 else ''}{pnl['pct_return']:.1f}%)  "
                 f"Today: {'+' if pnl['today_change_val']>=0 else ''}{pnl['today_change_pct']:.1f}%")

    # Always send a brief daily digest
    send_daily_digest(signal, state, day_number, anomalies, pnl=pnl)

    # Weekly summary (every Friday or every 7 days)
    today_dt = datetime.today()
    if today_dt.weekday() == FRIDAY_WEEKDAY or day_number % 7 == 0:
        send_weekly_summary(state, day_number)

    # Final report on day 30
    if day_number >= SHADOW_DAYS:
        state["completed"] = True
        report_path = generate_report(state)
        log.info(f"SHADOW MODE COMPLETE — 30-day report: {report_path}")
        # Queue completion email
        pending = LOGS_DIR / "pending_email.json"
        with open(pending, "w") as f:
            json.dump({
                "to":      ALERT_EMAIL,
                "subject": "[TRADING BOT] ✅ Shadow Mode Complete — 30-Day Report Ready",
                "body":    f"Shadow mode is complete.\n\nReport: {report_path}\n\n"
                           + Path(report_path).read_text(),
            }, f, indent=2)

    save_state(state)
    log.info(f"Day {day_number} complete. Next run: tomorrow.")
    return 2 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
