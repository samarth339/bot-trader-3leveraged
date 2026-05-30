"""
paper_trade.py — Headless Paper Trading Simulator (Phase 4)
=============================================================
Runs on GitHub Actions after market close (~4:00 PM ET daily).
Uses actual TQQQ closing prices from yfinance as MOC fill proxies.

No IB Gateway required. All state persists in git-committed files:
  logs/paper_portfolio.json  — current positions, NLV, peak equity
  logs/paper_trades.csv      — full trade history

The simulation mirrors ibkr/executor.py but replaces the IB connection
with yfinance price fetches and a JSON state file.

Safety checks run (IB-agnostic subset):
  ✓ Kill switch                       (logs/ibkr_kill.flag)
  ✓ Signal staleness                  (as_of_date must be today or last trading day)
  ✓ Double submission                 (last_trade_date in portfolio state)
  ✓ Portfolio drawdown halt           (≥ 35% DD activates kill switch)
  ✓ Trade frequency limit             (≥ 100 trades/year)
  ✓ Gap guard                         (from signal row — logged at signal time)
  ✓ VIX extreme override              (live VIX close ≥ 45 blocks BUY)
  ✓ Daily stop-loss                   (7% intraday loss forces full exit)

Usage:
    python3 paper_trade.py             # normal daily run
    python3 paper_trade.py --dry-run   # compute plan, no state update
    python3 paper_trade.py --status    # print portfolio and exit
    python3 paper_trade.py --reset     # clear today's execution flag (recovery)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from config.strategy_config import (
    EXECUTION_CONFIG,
    RISK_CONFIG,
    STRATEGY_A_CONFIG,
    STRATEGY_B_CONFIG,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "paper_trade.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("paper_trade")

# ── Paths ─────────────────────────────────────────────────────────────────────
SIGNAL_LOG       = LOG_DIR / "signal_history.csv"
PORTFOLIO_FILE   = LOG_DIR / "paper_portfolio.json"
TRADES_CSV       = LOG_DIR / "paper_trades.csv"
KILL_SWITCH_PATH = LOG_DIR / "ibkr_kill.flag"

# ── Constants ─────────────────────────────────────────────────────────────────
SEED_CAPITAL     = 10_000.00     # paper trading seed capital
SLIPPAGE_BPS     = EXECUTION_CONFIG["slippage_bps"]   # 10 bps round-trip
DRIFT_THRESHOLD  = RISK_CONFIG["alloc_drift_rebalance"]
DD_HALT          = RISK_CONFIG["max_drawdown_halt"]
DAILY_STOP       = RISK_CONFIG["daily_stop_loss"]
MAX_TRADES_YTD   = 100
VIX_EXTREME      = 45.0
EST              = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio state  (JSON file)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PORTFOLIO = {
    "tqqq_shares":      0,
    "cash":             SEED_CAPITAL,
    "nlv":              SEED_CAPITAL,
    "peak_equity":      SEED_CAPITAL,
    "seed_capital":     SEED_CAPITAL,
    "total_trades_ytd": 0,
    "last_trade_date":  None,
    "last_fill_price":  0.0,
    "inception_date":   str(date.today()),
}


def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            state = json.loads(PORTFOLIO_FILE.read_text())
            # Backfill any keys added after inception
            for k, v in DEFAULT_PORTFOLIO.items():
                state.setdefault(k, v)
            return state
        except Exception as exc:
            logger.warning(f"Could not load portfolio state: {exc} — using defaults")
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(state: dict) -> None:
    PORTFOLIO_FILE.write_text(json.dumps(state, indent=2, default=str))


# ══════════════════════════════════════════════════════════════════════════════
#  Trade history  (CSV)
# ══════════════════════════════════════════════════════════════════════════════

TRADE_COLUMNS = [
    "date", "regime", "action", "status",
    "tqqq_shares_before", "delta_shares", "tqqq_shares_after",
    "fill_price", "slippage_bps", "cash_before", "cash_after",
    "nlv_before", "nlv_after", "target_pct", "actual_pct",
    "gap_guard", "notes",
]


def _ensure_trades_csv() -> None:
    if not TRADES_CSV.exists():
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADES_CSV, index=False)


def append_trade(row: dict) -> None:
    _ensure_trades_csv()
    df = pd.read_csv(TRADES_CSV)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES_CSV, index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Price fetching  (handles yfinance MultiIndex — yfinance ≥ 1.3.0)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance 1.3+ MultiIndex columns to plain column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    return df


def fetch_close(ticker: str) -> Optional[float]:
    """Fetch today's closing price. Returns None if unavailable."""
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        data = _flatten(data)
        if data.empty:
            return None
        # Use the most recent bar's close
        price = float(data["Close"].iloc[-1])
        logger.info(f"Close price {ticker}: ${price:.2f}")
        return price
    except Exception as exc:
        logger.warning(f"Could not fetch {ticker} close: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Signal reading
# ══════════════════════════════════════════════════════════════════════════════

def read_signal() -> dict:
    """
    Read the most recent signal from signal_history.csv.
    Raises ValueError if signal is stale (not from today or the last trading day).
    """
    if not SIGNAL_LOG.exists():
        raise FileNotFoundError(
            f"Signal log not found: {SIGNAL_LOG} — run daily_signal.py first"
        )

    df = pd.read_csv(SIGNAL_LOG, parse_dates=["as_of_date"])
    if df.empty:
        raise ValueError("Signal log is empty — run daily_signal.py first")

    row         = df.iloc[-1].to_dict()
    signal_date = pd.Timestamp(row["as_of_date"]).date()
    today       = date.today()

    # Allow signal from up to 4 calendar days ago (covers long weekends)
    age_days = (today - signal_date).days
    if age_days > 4:
        raise ValueError(
            f"Signal is stale: as_of_date={signal_date}, today={today} "
            f"({age_days} days old). Run daily_signal.py to refresh."
        )

    if row.get("shadow", True):
        raise ValueError(
            "Signal has shadow=True — this is a shadow-mode signal. "
            "Run daily_signal.py without --shadow for Phase 4."
        )

    logger.info(
        f"Signal loaded: {signal_date}  regime={row['regime']}  "
        f"action={row['action']}  weight_a={row['weight_a']}  "
        f"weight_b={row['weight_b']}"
    )
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  Safety checks  (IB-agnostic subset of ibkr/safety_guard.py)
# ══════════════════════════════════════════════════════════════════════════════

class PaperSafetyGuard:

    def __init__(self, signal: dict, portfolio: dict):
        self.signal    = signal
        self.portfolio = portfolio
        self._daily_stop_triggered = False

    def run_all_checks(self) -> tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str).
        Checks run in priority order; first failure short-circuits.
        """
        checks = [
            ("kill_switch",     self._check_kill_switch),
            ("double_submit",   self._check_double_submission),
            ("portfolio_dd",    self._check_drawdown),
            ("trade_frequency", self._check_trade_frequency),
            ("daily_loss",      self._check_daily_loss),      # mutates signal
            ("gap_guard",       self._check_gap_guard),
            ("vix_extreme",     self._check_vix_extreme),
        ]
        for name, fn in checks:
            blocked, reason = fn()
            if blocked:
                logger.warning(f"Guard [{name}] BLOCKED: {reason}")
                return True, reason
            logger.debug(f"Guard [{name}] ✓")
        logger.info("All paper safety guards passed ✓")
        return False, ""

    # ── Guards ────────────────────────────────────────────────────────────────

    def _check_kill_switch(self) -> tuple[bool, str]:
        if KILL_SWITCH_PATH.exists():
            reason = KILL_SWITCH_PATH.read_text().strip()
            return True, f"Kill switch active — {reason}"
        return False, ""

    def _check_double_submission(self) -> tuple[bool, str]:
        last = self.portfolio.get("last_trade_date")
        if last and str(last)[:10] == str(date.today()):
            return True, f"Already executed today ({last})"
        return False, ""

    def _check_drawdown(self) -> tuple[bool, str]:
        nlv  = self.portfolio.get("nlv", SEED_CAPITAL)
        peak = self.portfolio.get("peak_equity", SEED_CAPITAL)
        if peak <= 0:
            return False, ""
        dd = (peak - nlv) / peak
        if dd >= DD_HALT:
            msg = (
                f"Portfolio drawdown {dd:.1%} ≥ halt threshold {DD_HALT:.1%} "
                f"(peak=${peak:,.0f}, NLV=${nlv:,.0f}) — kill switch activated"
            )
            KILL_SWITCH_PATH.write_text(msg)
            return True, msg
        if dd >= 0.30:
            logger.warning(f"Drawdown warning: {dd:.1%} (halt at {DD_HALT:.1%})")
        return False, ""

    def _check_trade_frequency(self) -> tuple[bool, str]:
        ytd = self.portfolio.get("total_trades_ytd", 0)
        if ytd >= MAX_TRADES_YTD:
            return True, (
                f"Trade count {ytd} ≥ {MAX_TRADES_YTD}/year limit. "
                "Frequent TQQQ trading destroys returns via 3x decay."
            )
        return False, ""

    def _check_daily_loss(self) -> tuple[bool, str]:
        """Non-blocking: mutates signal if daily loss exceeds threshold."""
        last_fill = self.portfolio.get("last_fill_price", 0.0)
        shares    = self.portfolio.get("tqqq_shares", 0)
        if last_fill <= 0 or shares <= 0:
            return False, ""
        try:
            price = fetch_close("TQQQ")
            if price is None:
                return False, ""
            daily_loss = (last_fill - price) / last_fill
            if daily_loss >= DAILY_STOP:
                logger.warning(
                    f"Daily stop-loss triggered ({daily_loss:.1%} ≥ {DAILY_STOP:.1%}) "
                    "— overriding to full exit"
                )
                self.signal["_daily_stop_triggered"] = True
                self._daily_stop_triggered = True
        except Exception as exc:
            logger.warning(f"Daily loss check skipped: {exc}")
        return False, ""   # never blocks, only mutates

    def _check_gap_guard(self) -> tuple[bool, str]:
        """Read gap_guard from the signal row (logged at signal-generation time)."""
        gap_triggered = str(self.signal.get("gap_guard", "")).lower() in ("true", "1")
        if not gap_triggered:
            return False, ""

        action   = str(self.signal.get("action", "HOLD"))
        weight_a = float(self.signal.get("weight_a", 0))
        is_buying = (
            action in ("INCREASE_A", "HOLD")
            and weight_a > 0
            and not self.signal.get("_daily_stop_triggered", False)
        )
        if is_buying:
            gap_pct = self.signal.get("gap_pct", "?")
            return True, (
                f"Gap guard triggered ({gap_pct}% open gap) — "
                "BUY orders blocked. 87% of >5% gap-down days close lower."
            )
        logger.warning("Gap guard triggered but action is SELL/exit — allowing")
        return False, ""

    def _check_vix_extreme(self) -> tuple[bool, str]:
        """Block BUY orders when live VIX ≥ 45."""
        try:
            vix = fetch_close("^VIX")
            if vix is None:
                return False, ""
            logger.info(f"VIX close: {vix:.1f}  (extreme threshold: {VIX_EXTREME})")
            if vix >= VIX_EXTREME:
                action   = self.signal.get("action", "HOLD")
                weight_a = float(self.signal.get("weight_a", 0))
                is_buying = (
                    action in ("INCREASE_A", "HOLD")
                    and weight_a > 0
                    and not self.signal.get("_daily_stop_triggered", False)
                )
                if is_buying:
                    return True, (
                        f"Live VIX={vix:.1f} ≥ {VIX_EXTREME} (extreme crash). "
                        "BUY orders blocked."
                    )
                logger.warning(f"VIX extreme ({vix:.1f}) but action is SELL — allowing")
        except Exception as exc:
            logger.warning(f"VIX check skipped: {exc}")
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  Allocation math  (mirrors position_reconciler.py without IB dependency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_target_pct(signal: dict) -> float:
    """Blended TQQQ allocation as fraction of NLV."""
    if signal.get("_daily_stop_triggered"):
        logger.warning("Daily stop override: target = 0.0% (full exit)")
        return 0.0
    weight_a  = float(signal.get("weight_a", 0.75))
    weight_b  = float(signal.get("weight_b", 0.25))
    max_pos_a = STRATEGY_A_CONFIG["max_position_pct"]
    max_pos_b = STRATEGY_B_CONFIG["max_position_pct"]
    blended   = weight_a * max_pos_a + weight_b * max_pos_b
    logger.info(
        f"Target allocation: {weight_a:.0%}×{max_pos_a:.0%} + "
        f"{weight_b:.0%}×{max_pos_b:.0%} = {blended:.1%}"
    )
    return blended


def compute_plan(
    signal: dict,
    portfolio: dict,
    tqqq_price: float,
) -> dict:
    """
    Compute the rebalancing plan.
    Returns a dict describing the trade to execute (or skip).
    """
    current_shares = portfolio["tqqq_shares"]
    cash           = portfolio["cash"]
    nlv            = current_shares * tqqq_price + cash
    target_pct     = compute_target_pct(signal)
    target_value   = nlv * target_pct
    target_shares  = math.floor(target_value / tqqq_price) if tqqq_price > 0 else 0
    current_pct    = (current_shares * tqqq_price) / nlv if nlv > 0 else 0.0
    drift_pct      = abs(target_pct - current_pct)
    delta_shares   = target_shares - current_shares

    # Drift gate: skip if already close enough
    within_tolerance = drift_pct < DRIFT_THRESHOLD

    logger.info(
        f"Plan: current={current_shares}sh ({current_pct:.1%})  "
        f"target={target_shares}sh ({target_pct:.1%})  "
        f"drift={drift_pct:.1%}  delta={delta_shares:+d}  NLV=${nlv:,.0f}"
    )

    if within_tolerance or delta_shares == 0:
        reason = (
            f"Within drift tolerance ({drift_pct:.1%} < {DRIFT_THRESHOLD:.1%})"
            if within_tolerance else "Delta is zero — already at target"
        )
        return {
            "proceed": False,
            "status": "no_action",
            "reason": reason,
            "delta_shares": 0,
            "target_shares": target_shares,
            "current_shares": current_shares,
            "target_pct": target_pct,
            "current_pct": current_pct,
            "nlv": nlv,
        }

    direction = "BUY" if delta_shares > 0 else "SELL"
    return {
        "proceed": True,
        "status": "pending",
        "direction": direction,
        "reason": f"{direction} {abs(delta_shares)} TQQQ ({current_pct:.1%} → {target_pct:.1%})",
        "delta_shares": delta_shares,
        "target_shares": target_shares,
        "current_shares": current_shares,
        "target_pct": target_pct,
        "current_pct": current_pct,
        "nlv": nlv,
    }


def execute_paper_fill(
    plan: dict,
    portfolio: dict,
    tqqq_price: float,
    slippage_bps: int = SLIPPAGE_BPS,
) -> dict:
    """
    Simulate a MOC fill at closing price ± slippage.
    Returns updated portfolio state.
    """
    delta    = plan["delta_shares"]
    factor   = slippage_bps / 10_000
    if delta > 0:   # BUY: pay slightly above close
        fill = round(tqqq_price * (1 + factor), 4)
    else:           # SELL: receive slightly below close
        fill = round(tqqq_price * (1 - factor), 4)

    cost           = delta * fill          # positive = cash out, negative = cash in
    new_shares     = portfolio["tqqq_shares"] + delta
    new_cash       = portfolio["cash"] - cost
    new_nlv        = new_shares * tqqq_price + new_cash
    new_peak       = max(portfolio["peak_equity"], new_nlv)
    new_ytd        = portfolio["total_trades_ytd"] + 1

    logger.info(
        f"Paper fill: {delta:+d} TQQQ @ ${fill:.2f}  "
        f"(close=${tqqq_price:.2f}  slippage={slippage_bps}bps)  "
        f"cost=${cost:+,.2f}  new_cash=${new_cash:,.2f}  NLV=${new_nlv:,.2f}"
    )

    updated = {
        **portfolio,
        "tqqq_shares":      new_shares,
        "cash":             round(new_cash, 2),
        "nlv":              round(new_nlv, 2),
        "peak_equity":      round(new_peak, 2),
        "total_trades_ytd": new_ytd,
        "last_trade_date":  str(date.today()),
        "last_fill_price":  fill,
    }
    return updated, fill


# ══════════════════════════════════════════════════════════════════════════════
#  Email alert  (mirrors executor._send_alert)
# ══════════════════════════════════════════════════════════════════════════════

def send_alert(subject: str, body: str) -> None:
    try:
        import send_email
        send_email.send_email(subject=subject, body=body)
        logger.debug(f"Alert sent: {subject}")
    except Exception as exc:
        logger.warning(f"Email alert failed (non-critical): {exc}")


def build_email(signal: dict, plan: dict, portfolio_after: dict,
                fill_price: float, blocked: bool, reason: str,
                dry_run: bool) -> tuple[str, str]:
    """Build subject + body for execution email."""
    today    = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    tag      = " [DRY RUN]" if dry_run else ""
    regime   = signal.get("regime", "?").upper()
    nlv      = portfolio_after.get("nlv", 0)
    delta    = plan.get("delta_shares", 0)
    status   = "BLOCKED" if blocked else plan.get("status", "?")

    if blocked:
        subject = f"[PAPER{tag}] BLOCKED — {reason[:50]}  {signal.get('as_of_date','')}"
    elif plan.get("status") == "no_action":
        subject = f"[PAPER{tag}] HOLD {regime}  ${nlv:,.0f}  {signal.get('as_of_date','')}"
    else:
        direction = plan.get("direction", "?")
        subject   = (f"[PAPER{tag}] {direction} {abs(delta)} TQQQ "
                     f"({regime})  ${nlv:,.0f}  {signal.get('as_of_date','')}")

    body = "\n".join([
        f"Paper Trade Report — {today}{tag}",
        "",
        f"Regime   : {regime}",
        f"Action   : {signal.get('action','—')}",
        f"Status   : {status}",
        "",
        f"Delta    : {delta:+d} shares TQQQ",
        f"Fill     : ${fill_price:.2f}" if fill_price else "Fill     : n/a",
        "",
        f"NLV      : ${nlv:,.2f}",
        f"Shares   : {portfolio_after.get('tqqq_shares',0)} TQQQ",
        f"Cash     : ${portfolio_after.get('cash',0):,.2f}",
        f"Return   : {(nlv - portfolio_after.get('seed_capital', nlv)) / portfolio_after.get('seed_capital', nlv) * 100:+.2f}%",
        "",
        f"Reason   : {reason or plan.get('reason','—')}",
        "",
        "Logs: logs/paper_trade.log",
        "State: logs/paper_portfolio.json",
        "Trades: logs/paper_trades.csv",
    ])
    return subject, body


# ══════════════════════════════════════════════════════════════════════════════
#  Main execution flow
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> bool:
    """
    Full paper execution flow. Returns True on success / clean no-action.
    """
    now = datetime.now(EST).strftime("%Y-%m-%d %H:%M %Z")
    logger.info("=" * 65)
    logger.info(f"  Paper Trade  |  {'DRY RUN  ' if dry_run else 'LIVE PAPER'}  |  {now}")
    logger.info("=" * 65)

    # ── Step 1: Read signal ────────────────────────────────────────────────
    try:
        signal = read_signal()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Signal read failed: {exc}")
        send_alert(
            subject="[PAPER] BLOCKED — signal error",
            body=f"Could not read today's signal:\n{exc}\n\nRun: python3 daily_signal.py",
        )
        return False

    # ── Step 2: Load portfolio state ──────────────────────────────────────
    portfolio = load_portfolio()
    logger.info(
        f"Portfolio: {portfolio['tqqq_shares']} TQQQ  "
        f"cash=${portfolio['cash']:,.2f}  NLV=${portfolio['nlv']:,.2f}"
    )

    # ── Step 3: Fetch TQQQ closing price ──────────────────────────────────
    tqqq_price = fetch_close("TQQQ")
    if tqqq_price is None:
        logger.error("Cannot fetch TQQQ price — aborting to prevent sizing errors")
        send_alert(
            subject="[PAPER] BLOCKED — TQQQ price unavailable",
            body="yfinance could not return today's TQQQ closing price.",
        )
        return False

    # Update NLV with current price before guards run
    portfolio["nlv"] = round(portfolio["tqqq_shares"] * tqqq_price + portfolio["cash"], 2)

    # ── Step 4: Safety guards ──────────────────────────────────────────────
    guard            = PaperSafetyGuard(signal=signal, portfolio=portfolio)
    blocked, reason  = guard.run_all_checks()

    if blocked:
        send_alert(*build_email(signal, {"delta_shares": 0, "status": "blocked"},
                                portfolio, 0, True, reason, dry_run))
        return False

    # ── Step 5: Compute rebalancing plan ──────────────────────────────────
    plan = compute_plan(signal, portfolio, tqqq_price)
    logger.info(f"Plan: {plan['reason']}")

    # ── Step 6: Execute fill (or skip) ────────────────────────────────────
    fill_price     = 0.0
    portfolio_after = portfolio.copy()

    if plan["proceed"] and not dry_run:
        portfolio_after, fill_price = execute_paper_fill(plan, portfolio, tqqq_price)

        # Append trade record
        append_trade({
            "date":               str(date.today()),
            "regime":             signal.get("regime", "?"),
            "action":             signal.get("action", "?"),
            "status":             "executed",
            "tqqq_shares_before": portfolio["tqqq_shares"],
            "delta_shares":       plan["delta_shares"],
            "tqqq_shares_after":  portfolio_after["tqqq_shares"],
            "fill_price":         fill_price,
            "slippage_bps":       SLIPPAGE_BPS,
            "cash_before":        portfolio["cash"],
            "cash_after":         portfolio_after["cash"],
            "nlv_before":         portfolio["nlv"],
            "nlv_after":          portfolio_after["nlv"],
            "target_pct":         round(plan["target_pct"] * 100, 2),
            "actual_pct":         round(plan["target_pct"] * 100, 2),
            "gap_guard":          signal.get("gap_guard", False),
            "notes":              plan["reason"],
        })
    elif not plan["proceed"] and not dry_run:
        # Record the no-action day too
        append_trade({
            "date":               str(date.today()),
            "regime":             signal.get("regime", "?"),
            "action":             signal.get("action", "?"),
            "status":             "no_action",
            "tqqq_shares_before": portfolio["tqqq_shares"],
            "delta_shares":       0,
            "tqqq_shares_after":  portfolio["tqqq_shares"],
            "fill_price":         "",
            "slippage_bps":       0,
            "cash_before":        portfolio["cash"],
            "cash_after":         portfolio["cash"],
            "nlv_before":         portfolio["nlv"],
            "nlv_after":          portfolio["nlv"],
            "target_pct":         round(plan["target_pct"] * 100, 2),
            "actual_pct":         round(plan["current_pct"] * 100, 2),
            "gap_guard":          signal.get("gap_guard", False),
            "notes":              plan["reason"],
        })

    # ── Step 7: Persist state ──────────────────────────────────────────────
    if not dry_run:
        if plan["proceed"]:
            save_portfolio(portfolio_after)
        else:
            # Update NLV even on no-action (price moved)
            portfolio["nlv"] = round(portfolio["tqqq_shares"] * tqqq_price + portfolio["cash"], 2)
            portfolio["peak_equity"] = max(portfolio["peak_equity"], portfolio["nlv"])
            save_portfolio(portfolio)

    # ── Step 8: Log summary ────────────────────────────────────────────────
    _log_summary(signal, plan, portfolio_after, fill_price, dry_run)

    # ── Step 9: Email ──────────────────────────────────────────────────────
    send_alert(*build_email(signal, plan, portfolio_after, fill_price, False, "", dry_run))

    return True


def _log_summary(signal, plan, portfolio, fill_price, dry_run):
    tag = "[DRY RUN]" if dry_run else ""
    logger.info("-" * 65)
    logger.info(f"  PAPER EXECUTION SUMMARY  {tag}")
    logger.info(f"  Regime:        {signal.get('regime','?').upper()}")
    logger.info(f"  Action:        {signal.get('action','?')}")
    logger.info(f"  Status:        {plan.get('status','?')}")
    logger.info(f"  Delta:         {plan.get('delta_shares',0):+d} shares")
    if fill_price:
        logger.info(f"  Fill price:    ${fill_price:.2f}")
    logger.info(f"  TQQQ shares:   {portfolio.get('tqqq_shares',0)}")
    logger.info(f"  Cash:          ${portfolio.get('cash',0):,.2f}")
    logger.info(f"  NLV:           ${portfolio.get('nlv',0):,.2f}")
    ret = (portfolio.get("nlv",0) - portfolio.get("seed_capital",0)) / portfolio.get("seed_capital",1) * 100
    logger.info(f"  Return:        {ret:+.2f}%")
    logger.info("-" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  Status command
# ══════════════════════════════════════════════════════════════════════════════

def print_status():
    portfolio = load_portfolio()
    ks_active = KILL_SWITCH_PATH.exists()

    print()
    print("═" * 55)
    print("  Paper Trade — Portfolio Status")
    print("═" * 55)
    print(f"  Kill switch:        {'ACTIVE ⛔' if ks_active else 'OFF ✓'}")
    print(f"  TQQQ shares:        {portfolio['tqqq_shares']}")
    print(f"  Cash:               ${portfolio['cash']:,.2f}")
    print(f"  NLV:                ${portfolio['nlv']:,.2f}")
    print(f"  Seed capital:       ${portfolio['seed_capital']:,.2f}")
    ret = (portfolio['nlv'] - portfolio['seed_capital']) / portfolio['seed_capital'] * 100
    print(f"  Return:             {ret:+.2f}%")
    print(f"  Peak equity:        ${portfolio['peak_equity']:,.2f}")
    dd = (portfolio['peak_equity'] - portfolio['nlv']) / portfolio['peak_equity'] if portfolio['peak_equity'] > 0 else 0
    print(f"  Drawdown:           {dd:.1%}")
    print(f"  Trades YTD:         {portfolio['total_trades_ytd']}")
    print(f"  Last trade:         {portfolio['last_trade_date'] or 'never'}")
    print(f"  Last fill:          ${portfolio['last_fill_price']:.2f}" if portfolio['last_fill_price'] else "  Last fill:          —")
    print(f"  Inception:          {portfolio['inception_date']}")
    print("═" * 55)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Headless paper trading simulator (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 paper_trade.py                 # daily run\n"
            "  python3 paper_trade.py --dry-run       # compute plan only\n"
            "  python3 paper_trade.py --status        # print portfolio\n"
            "  python3 paper_trade.py --reset         # clear today's execution flag\n"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute plan and log — do NOT update portfolio state")
    parser.add_argument("--status",  action="store_true",
                        help="Print current portfolio state and exit")
    parser.add_argument("--reset",   action="store_true",
                        help="Clear today's execution flag (use after failed run recovery)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.reset:
        portfolio = load_portfolio()
        portfolio["last_trade_date"] = None
        save_portfolio(portfolio)
        print("Execution flag cleared — ready to re-run today")
        return

    success = run(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
