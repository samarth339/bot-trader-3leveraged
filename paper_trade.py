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
  ✓ Signal staleness                  (as_of_date must be today or last trading day)
  ✓ Double submission                 (last_trade_date in portfolio state)   [hard block]
  ✓ Kill switch                       (logs/ibkr_kill.flag)                  [flatten, then freeze buys]
  ✓ Portfolio drawdown halt           (≥ 35% DD activates kill switch)       [flatten, then freeze buys]
  ✓ Trade frequency limit             (≥ 100 trades/year)                    [blocks BUYs only; warns at 50]
  ✓ Crash-day check                   (TQQQ ≤ −7% vs prev close)             [blocks BUYs only]
  ✓ Gap guard                         (from signal row — logged at signal time) [blocks BUYs only]
  ✓ VIX extreme override              (live VIX close ≥ 45)                  [blocks BUYs only]

Position sizing comes from the replayed per-strategy exposure state
(exposure_a/exposure_b in the signal row): target = wa×exp_a + wb×exp_b.
This matches what the validated backtest holds, including its exits.

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


def fetch_recent_closes(ticker: str, n: int = 2) -> list:
    """Fetch the n most recent daily closes (oldest first). Empty list on failure."""
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        data = _flatten(data)
        if data.empty:
            return []
        closes = [float(c) for c in data["Close"].iloc[-n:]]
        logger.info(f"Recent closes {ticker}: " + "  ".join(f"${c:.2f}" for c in closes))
        return closes
    except Exception as exc:
        logger.warning(f"Could not fetch {ticker} closes: {exc}")
        return []


def fetch_close(ticker: str) -> Optional[float]:
    """Fetch today's closing price. Returns None if unavailable."""
    closes = fetch_recent_closes(ticker, n=1)
    return closes[-1] if closes else None


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
    """
    Three guard outcomes (checked in priority order):

      hard block      → no trade at all today (double submission, kill switch
                        with nothing left to flatten)
      force flatten   → sets signal["_force_flatten"]; the plan becomes a full
                        exit (kill switch / DD halt while still holding shares —
                        a halt must REDUCE leveraged exposure, never freeze it)
      buy block       → collected in self.buy_block_reasons; applied only if
                        the computed plan is a BUY (crash day, gap guard,
                        VIX extreme, trade-frequency cap). Sells always pass.
    """

    def __init__(self, signal: dict, portfolio: dict,
                 tqqq_closes: Optional[list] = None):
        self.signal            = signal
        self.portfolio         = portfolio
        self.tqqq_closes       = tqqq_closes or []   # [prev_close, last_close]
        self.buy_block_reasons = []

    def run_all_checks(self) -> tuple[bool, str]:
        """
        Returns (blocked: bool, reason: str) for HARD blocks only.
        Buy-only blocks accumulate in self.buy_block_reasons.
        """
        checks = [
            ("double_submit",   self._check_double_submission),
            ("kill_switch",     self._check_kill_switch),
            ("portfolio_dd",    self._check_drawdown),
            ("trade_frequency", self._check_trade_frequency),
            ("crash_day",       self._check_crash_day),
            ("gap_guard",       self._check_gap_guard),
            ("vix_extreme",     self._check_vix_extreme),
        ]
        for name, fn in checks:
            blocked, reason = fn()
            if blocked:
                logger.warning(f"Guard [{name}] BLOCKED: {reason}")
                return True, reason
            logger.debug(f"Guard [{name}] ✓")
        if self.buy_block_reasons:
            logger.warning(
                f"BUY orders blocked today ({len(self.buy_block_reasons)} guard(s)): "
                + " | ".join(self.buy_block_reasons)
            )
        logger.info("Paper safety guards passed ✓ (hard blocks)")
        return False, ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _flatten_or_block(self, reason: str) -> tuple[bool, str]:
        """Holding shares → force a full exit; flat → hard block (buys frozen)."""
        if self.portfolio.get("tqqq_shares", 0) > 0:
            logger.warning(f"{reason} — flattening position (full exit at close)")
            self.signal["_force_flatten"] = True
            self.signal["_force_flatten_reason"] = reason
            return False, ""
        return True, f"{reason} — no position held, all trading frozen"

    # ── Guards ────────────────────────────────────────────────────────────────

    def _check_double_submission(self) -> tuple[bool, str]:
        last = self.portfolio.get("last_trade_date")
        if last and str(last)[:10] == str(date.today()):
            return True, f"Already executed today ({last})"
        return False, ""

    def _check_kill_switch(self) -> tuple[bool, str]:
        if KILL_SWITCH_PATH.exists():
            reason = KILL_SWITCH_PATH.read_text().strip()
            return self._flatten_or_block(f"Kill switch active — {reason}")
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
            return self._flatten_or_block(msg)
        if dd >= 0.30:
            logger.warning(f"Drawdown warning: {dd:.1%} (halt at {DD_HALT:.1%})")
        return False, ""

    def _check_trade_frequency(self) -> tuple[bool, str]:
        """Buy-block only — a frequency cap must never stop risk-reducing sells."""
        ytd = self.portfolio.get("total_trades_ytd", 0)
        if ytd >= MAX_TRADES_YTD:
            self.buy_block_reasons.append(
                f"Trade count {ytd} ≥ {MAX_TRADES_YTD}/year limit — BUYs blocked "
                "(frequent TQQQ trading destroys returns via 3x decay)"
            )
        elif ytd >= MAX_TRADES_YTD // 2:
            logger.warning(
                f"Trade frequency warning: {ytd}/{MAX_TRADES_YTD} trades this year — "
                "investigate churn before the cap blocks buying"
            )
        return False, ""

    def _check_crash_day(self) -> tuple[bool, str]:
        """
        Same-day crash protection: TQQQ down ≥ daily_stop_loss vs the previous
        close → block BUYs today. Exits are handled by the replayed strategy
        state on the next signal (mirrors the backtest engine's same-bar stop
        with one bar of lag). This replaces the old last-fill-relative stop,
        which sold entire positions at the close and re-bought full size the
        next day — two losing round trips in the first week of paper trading.
        """
        if len(self.tqqq_closes) < 2:
            logger.warning("Crash-day check skipped: insufficient price history")
            return False, ""
        prev_close, last_close = self.tqqq_closes[-2], self.tqqq_closes[-1]
        if prev_close <= 0:
            return False, ""
        day_change = (last_close - prev_close) / prev_close
        logger.info(f"Crash-day check: TQQQ {day_change:+.1%} vs prev close "
                    f"(buy-block at {-DAILY_STOP:.0%})")
        if day_change <= -DAILY_STOP:
            self.buy_block_reasons.append(
                f"TQQQ {day_change:+.1%} today (≥ {DAILY_STOP:.0%} drop) — "
                "no BUYs into a crash; strategy state re-evaluates tomorrow"
            )
        return False, ""

    def _check_gap_guard(self) -> tuple[bool, str]:
        """Read gap_guard from the signal row (logged at signal-generation time)."""
        if str(self.signal.get("gap_guard", "")).lower() in ("true", "1"):
            gap_pct = self.signal.get("gap_pct", "?")
            self.buy_block_reasons.append(
                f"Gap guard triggered ({gap_pct}% open gap) — "
                "87% of >5% gap-down days close lower"
            )
        return False, ""

    def _check_vix_extreme(self) -> tuple[bool, str]:
        """Block BUY orders when live VIX ≥ 45."""
        try:
            vix = fetch_close("^VIX")
            if vix is None:
                return False, ""
            logger.info(f"VIX close: {vix:.1f}  (extreme threshold: {VIX_EXTREME})")
            if vix >= VIX_EXTREME:
                self.buy_block_reasons.append(
                    f"Live VIX={vix:.1f} ≥ {VIX_EXTREME} (extreme crash)"
                )
        except Exception as exc:
            logger.warning(f"VIX check skipped: {exc}")
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  Allocation math  (mirrors position_reconciler.py without IB dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_exposure(signal: dict, key: str) -> Optional[float]:
    """Read an exposure field from the signal row; None when missing/blank."""
    raw = signal.get(key)
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or val < 0:
        return None
    return val


def compute_target_pct(signal: dict) -> float:
    """
    Blended TQQQ allocation as fraction of NLV.

    Primary path: weight_a×exposure_a + weight_b×exposure_b, where the
    exposures are the replayed per-strategy states (backtester/exposure_replay
    via daily_signal.py). This is the allocation the validated backtest
    actually holds — including its MA/VIX exits, vol scaling, and crash brake.

    Fallback (exposure fields missing): weight×max_position_pct. That floors
    exposure at ~66% TQQQ even in high_vol — log loudly, this is the legacy
    behavior that produced 60%+ drawdowns in replay.
    """
    if signal.get("_force_flatten"):
        logger.warning(
            f"Force-flatten override: target = 0.0% (full exit) — "
            f"{signal.get('_force_flatten_reason', 'risk halt')}"
        )
        return 0.0

    weight_a = float(signal.get("weight_a", 0.75))
    weight_b = float(signal.get("weight_b", 0.25))

    exp_a = _parse_exposure(signal, "exposure_a")
    exp_b = _parse_exposure(signal, "exposure_b")

    if exp_a is not None and exp_b is not None:
        blended = weight_a * exp_a + weight_b * exp_b
        logger.info(
            f"Target allocation (exposure state): {weight_a:.0%}×{exp_a:.1%} + "
            f"{weight_b:.0%}×{exp_b:.1%} = {blended:.1%}"
        )
        return blended

    max_pos_a = STRATEGY_A_CONFIG["max_position_pct"]
    max_pos_b = STRATEGY_B_CONFIG["max_position_pct"]
    blended   = weight_a * max_pos_a + weight_b * max_pos_b
    logger.error(
        "Signal row has no exposure_a/exposure_b — falling back to "
        f"max-position caps: {weight_a:.0%}×{max_pos_a:.0%} + "
        f"{weight_b:.0%}×{max_pos_b:.0%} = {blended:.1%}. "
        "This OVER-ALLOCATES whenever the strategies are de-risked; "
        "re-run daily_signal.py."
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

    # Drift gate: skip if already close enough.
    # A force-flatten (kill switch / DD halt) always sells to zero — never
    # leave residual leveraged shares behind a halt.
    force_flatten    = bool(signal.get("_force_flatten")) and current_shares > 0
    within_tolerance = drift_pct < DRIFT_THRESHOLD and not force_flatten
    if force_flatten:
        target_shares, delta_shares = 0, -current_shares

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
    Full paper execution flow.

    Returns True for any cleanly-handled outcome — a trade, a no-action day, a
    guard hard-block (kill switch / DD halt), or a buy-block (gap/crash/VIX/
    frequency). These are NORMAL: the automation worked and decided correctly,
    so the workflow stays green and still commits the audit record.

    Returns False ONLY on a genuine operational error (signal unreadable, price
    feed down) — those should turn the workflow red so a human investigates.
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

    # Trade counter is per calendar year — reset on year rollover
    last_trade = str(portfolio.get("last_trade_date") or "")
    if last_trade[:4] and last_trade[:4] != str(date.today().year):
        logger.info(f"New calendar year — resetting trade counter "
                    f"(was {portfolio.get('total_trades_ytd', 0)})")
        portfolio["total_trades_ytd"] = 0

    logger.info(
        f"Portfolio: {portfolio['tqqq_shares']} TQQQ  "
        f"cash=${portfolio['cash']:,.2f}  NLV=${portfolio['nlv']:,.2f}"
    )

    # ── Step 3: Fetch TQQQ closing prices (prev + last, for crash check) ──
    tqqq_closes = fetch_recent_closes("TQQQ", n=2)
    if not tqqq_closes:
        logger.error("Cannot fetch TQQQ price — aborting to prevent sizing errors")
        send_alert(
            subject="[PAPER] BLOCKED — TQQQ price unavailable",
            body="yfinance could not return today's TQQQ closing price.",
        )
        return False
    tqqq_price = tqqq_closes[-1]

    # Update NLV with current price before guards run
    portfolio["nlv"] = round(portfolio["tqqq_shares"] * tqqq_price + portfolio["cash"], 2)

    # ── Step 4: Safety guards ──────────────────────────────────────────────
    guard            = PaperSafetyGuard(signal=signal, portfolio=portfolio,
                                        tqqq_closes=tqqq_closes)
    blocked, reason  = guard.run_all_checks()

    if blocked:
        # Persist the no-action record so the halt is visible in the audit log.
        if not dry_run:
            append_trade({
                "date":               str(date.today()),
                "regime":             signal.get("regime", "?"),
                "action":             signal.get("action", "?"),
                "status":             "blocked",
                "tqqq_shares_before": portfolio["tqqq_shares"],
                "delta_shares":       0,
                "tqqq_shares_after":  portfolio["tqqq_shares"],
                "fill_price":         "",
                "slippage_bps":       0,
                "cash_before":        portfolio["cash"],
                "cash_after":         portfolio["cash"],
                "nlv_before":         portfolio["nlv"],
                "nlv_after":          portfolio["nlv"],
                "target_pct":         "",
                "actual_pct":         "",
                "gap_guard":          signal.get("gap_guard", False),
                "notes":              f"BLOCKED: {reason}",
            })
            portfolio["peak_equity"] = max(portfolio["peak_equity"], portfolio["nlv"])
            save_portfolio(portfolio)
        send_alert(*build_email(signal, {"delta_shares": 0, "status": "blocked"},
                                portfolio, 0, True, reason, dry_run))
        return True   # handled cleanly — halt is not an automation failure

    # ── Step 5: Compute rebalancing plan ──────────────────────────────────
    plan = compute_plan(signal, portfolio, tqqq_price)
    logger.info(f"Plan: {plan['reason']}")

    # Buy-only guards (crash day, gap guard, VIX extreme, trade frequency):
    # applied to the PLAN, not the action heuristics — sells always proceed.
    if plan["proceed"] and plan.get("direction") == "BUY" and guard.buy_block_reasons:
        reason = " | ".join(guard.buy_block_reasons)
        logger.warning(f"BUY plan blocked: {reason}")
        if not dry_run:
            append_trade({
                "date":               str(date.today()),
                "regime":             signal.get("regime", "?"),
                "action":             signal.get("action", "?"),
                "status":             "buy_blocked",
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
                "notes":              f"BUY blocked: {reason}",
            })
            portfolio["peak_equity"] = max(portfolio["peak_equity"], portfolio["nlv"])
            save_portfolio(portfolio)
        send_alert(*build_email(signal, {"delta_shares": 0, "status": "buy_blocked"},
                                portfolio, 0, True, reason, dry_run))
        return True   # handled cleanly — a buy-block is the system working

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
#  Demo mode  — full simulation in a temp directory, no production files touched
# ══════════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """
    Execute the complete paper trading simulation against real live data,
    but redirect ALL file I/O to a temporary directory.

    Production files left completely untouched:
      - logs/paper_portfolio.json  (not read, not written)
      - logs/paper_trades.csv      (not read, not written)
      - logs/ibkr_kill.flag        (not read, not written)

    What the demo does:
      1. Reads today's signal from signal_history.csv  (read-only, unchanged)
      2. Initialises a fresh $10,000 Day-1 portfolio   (temp dir only)
      3. Fetches TQQQ & VIX live prices from yfinance
      4. Runs all 7 safety guards
      5. Computes and executes the simulated fill
      6. Prints a rich side-by-side before/after report
      7. Cleans up temp dir → production state unchanged
    """
    import tempfile, shutil

    global PORTFOLIO_FILE, TRADES_CSV, KILL_SWITCH_PATH

    # ── Save production paths ──────────────────────────────────────────────
    _orig_portfolio   = PORTFOLIO_FILE
    _orig_trades      = TRADES_CSV
    _orig_kill_switch = KILL_SWITCH_PATH

    tmp_dir = Path(tempfile.mkdtemp(prefix="paper_trade_demo_"))
    try:
        # ── Redirect all file I/O to temp dir ─────────────────────────────
        PORTFOLIO_FILE   = tmp_dir / "paper_portfolio.json"
        TRADES_CSV       = tmp_dir / "paper_trades.csv"
        KILL_SWITCH_PATH = tmp_dir / "ibkr_kill.flag"

        # ── Seed a fresh Day-1 portfolio ───────────────────────────────────
        demo_portfolio = DEFAULT_PORTFOLIO.copy()
        save_portfolio(demo_portfolio)
        _ensure_trades_csv()

        W = 62
        print()
        print("╔" + "═" * W + "╗")
        print(f"║{'  📋  PAPER TRADE — DEMO MODE':^{W}}║")
        print(f"║{'  Full simulation · real prices · temp files only':^{W}}║")
        print("╠" + "═" * W + "╣")
        print(f"║{'  Production files will NOT be touched':^{W}}║")
        print("╚" + "═" * W + "╝")
        print()

        # ── Step 1: Signal ─────────────────────────────────────────────────
        try:
            signal = read_signal()
        except (FileNotFoundError, ValueError) as exc:
            print(f"✗ Signal read failed: {exc}")
            return

        # ── Step 2: Load demo portfolio ────────────────────────────────────
        portfolio = load_portfolio()

        # ── Step 3: Live prices ────────────────────────────────────────────
        print("Fetching live prices …")
        tqqq_closes = fetch_recent_closes("TQQQ", n=2)
        tqqq_price  = tqqq_closes[-1] if tqqq_closes else None
        vix_price   = fetch_close("^VIX")
        if tqqq_price is None:
            print("✗ Cannot fetch TQQQ price — aborting demo")
            return
        portfolio["nlv"] = round(
            portfolio["tqqq_shares"] * tqqq_price + portfolio["cash"], 2
        )

        # ── Step 4: Safety guards ──────────────────────────────────────────
        guard           = PaperSafetyGuard(signal=signal, portfolio=portfolio,
                                           tqqq_closes=tqqq_closes)
        blocked, reason = guard.run_all_checks()

        # ── Step 5: Compute plan ───────────────────────────────────────────
        plan = compute_plan(signal, portfolio, tqqq_price) if not blocked else {
            "proceed": False, "status": "blocked", "delta_shares": 0,
            "target_pct": 0, "current_pct": 0, "nlv": portfolio["nlv"],
            "reason": reason,
        }

        # Buy-only guards apply to the plan (sells always proceed)
        if plan["proceed"] and plan.get("direction") == "BUY" and guard.buy_block_reasons:
            blocked, reason = True, " | ".join(guard.buy_block_reasons)
            plan = {
                "proceed": False, "status": "buy_blocked", "delta_shares": 0,
                "target_pct": plan["target_pct"], "current_pct": plan["current_pct"],
                "nlv": portfolio["nlv"], "reason": reason,
            }

        # ── Step 6: Execute fill ───────────────────────────────────────────
        fill_price      = 0.0
        portfolio_after = portfolio.copy()

        if plan["proceed"] and not blocked:
            portfolio_after, fill_price = execute_paper_fill(
                plan, portfolio, tqqq_price
            )
            append_trade({
                "date":               str(date.today()),
                "regime":             signal.get("regime", "?"),
                "action":             signal.get("action", "?"),
                "status":             "executed (DEMO)",
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
                "notes":              "[DEMO] " + plan["reason"],
            })

        # ── Step 7: Rich report ────────────────────────────────────────────
        regime  = signal.get("regime", "?").upper()
        action  = signal.get("action", "?")
        wa      = float(signal.get("weight_a", 0))
        wb      = float(signal.get("weight_b", 0))
        gap_raw = signal.get("gap_pct", "")
        gap_str = (f"{float(gap_raw):+.2f}%" if gap_raw not in ("", None, "nan") else "n/a")
        gap_triggered = str(signal.get("gap_guard", "")).lower() in ("true", "1")

        direction  = plan.get("direction", "—")
        delta      = plan.get("delta_shares", 0)
        target_pct = plan.get("target_pct", 0)
        status     = "BLOCKED" if blocked else plan.get("status", "?")

        ret_before = 0.0
        ret_after  = (
            (portfolio_after["nlv"] - portfolio_after["seed_capital"])
            / portfolio_after["seed_capital"] * 100
        ) if plan["proceed"] else 0.0

        slippage_cost = abs(fill_price - tqqq_price) * abs(delta) if fill_price else 0

        print()
        print("┌─ TODAY'S SIGNAL " + "─" * 45)
        print(f"│  Date       : {signal.get('as_of_date', '?')}")
        print(f"│  Regime     : {regime}")
        print(f"│  Action     : {action}")
        print(f"│  Weights    : {wa:.0%} Strategy-A  /  {wb:.0%} Strategy-B")
        print(f"│  QQQ (T-1)  : ${float(signal.get('qqq_price', 0)):.2f}  "
              f"({float(signal.get('pct_vs_sma', 0)):+.2f}% vs SMA-130)")
        print(f"│  VIX 5d avg : {signal.get('vix_signal', '?')}")
        print(f"│  Gap guard  : {'🚫 TRIGGERED' if gap_triggered else '✅ clear'}  ({gap_str})")
        print()
        print("┌─ SAFETY GUARDS " + "─" * 44)
        if blocked:
            print(f"│  ✗ BLOCKED — {reason}")
        else:
            print("│  ✓ All 7 guards passed")
        print()
        print("┌─ EXECUTION PLAN " + "─" * 43)
        print(f"│  Status     : {status.upper()}")
        if plan["proceed"]:
            print(f"│  Direction  : {direction}")
            print(f"│  Shares     : {delta:+d} TQQQ")
            print(f"│  Fill price : ${fill_price:.4f}  "
                  f"(close ${tqqq_price:.2f} + {SLIPPAGE_BPS} bps slippage)")
            print(f"│  Slip cost  : ${slippage_cost:.2f}  ({SLIPPAGE_BPS} bps × {abs(delta)} shares)")
            print(f"│  Target %   : {target_pct:.1%} of NLV")
        else:
            print(f"│  Reason     : {plan.get('reason', reason)}")
        print()
        print("┌─ PORTFOLIO BEFORE → AFTER " + "─" * 33)
        print(f"│  {'':30} {'BEFORE':>10}   {'AFTER':>10}")
        print(f"│  {'─'*52}")
        print(f"│  {'TQQQ shares':30} {portfolio['tqqq_shares']:>10}   "
              f"{portfolio_after['tqqq_shares']:>10}")
        print(f"│  {'Cash':30} ${portfolio['cash']:>9,.2f}   "
              f"${portfolio_after['cash']:>9,.2f}")
        print(f"│  {'NLV':30} ${portfolio['nlv']:>9,.2f}   "
              f"${portfolio_after['nlv']:>9,.2f}")
        print(f"│  {'TQQQ allocation':30} {portfolio['tqqq_shares'] * tqqq_price / portfolio['nlv']:>9.1%}   "
              f"{portfolio_after['tqqq_shares'] * tqqq_price / portfolio_after['nlv']:>9.1%}"
              if portfolio_after["nlv"] > 0 else "")
        print(f"│  {'Return vs $10K seed':30} {ret_before:>+9.2f}%   {ret_after:>+9.2f}%")
        print()

        # ── Show temp trade CSV ────────────────────────────────────────────
        if plan["proceed"]:
            import csv as _csv
            trades = list(_csv.DictReader(open(TRADES_CSV)))
            if trades:
                t = trades[-1]
                print("┌─ TRADE RECORD (what would be written to paper_trades.csv) " + "─" * 1)
                for k, v in t.items():
                    if v not in ("", None):
                        print(f"│  {k:<25} {v}")
                print()

        # ── Email preview ──────────────────────────────────────────────────
        subject, body = build_email(
            signal, plan, portfolio_after, fill_price, blocked, reason, False
        )
        print("┌─ EMAIL THAT WOULD BE SENT " + "─" * 33)
        print(f"│  Subject: {subject}")
        print("│")
        for line in body.splitlines()[:12]:
            print(f"│  {line}")
        print("│  …")
        print()

        print("╔" + "═" * W + "╗")
        print(f"║{'  ✅  DEMO COMPLETE':^{W}}║")
        print(f"║{'  Production files untouched:':^{W}}║")
        print(f"║{'  logs/paper_portfolio.json  ← still clean $10,000':^{W}}║")
        print(f"║{'  logs/paper_trades.csv      ← still empty':^{W}}║")
        print("╠" + "═" * W + "╣")
        print(f"║{'  Run  python3 paper_trade.py --reset-portfolio':^{W}}║")
        print(f"║{'  to confirm clean state before Day 1, then let':^{W}}║")
        print(f"║{'  GitHub Actions handle the real first execution.':^{W}}║")
        print("╚" + "═" * W + "╝")
        print()

    finally:
        # ── Always restore production paths ───────────────────────────────
        PORTFOLIO_FILE   = _orig_portfolio
        TRADES_CSV       = _orig_trades
        KILL_SWITCH_PATH = _orig_kill_switch
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Reset portfolio  — wipe back to clean Day-1 state
# ══════════════════════════════════════════════════════════════════════════════

def reset_portfolio() -> None:
    """
    Restore paper_portfolio.json and paper_trades.csv to the clean
    Day-1 state ($10,000 seed, 0 shares, no trades).

    Safe to run at any time — only modifies the two log files.
    """
    import csv as _csv

    clean = DEFAULT_PORTFOLIO.copy()
    clean["inception_date"] = str(date.today())

    save_portfolio(clean)

    # Rewrite trades CSV with header only
    with open(TRADES_CSV, "w", newline="") as f:
        _csv.DictWriter(f, fieldnames=TRADE_COLUMNS).writeheader()

    # Clear kill switch if present (safety)
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()
        print("⚠️  Kill switch was active — cleared.")

    print()
    print("╔" + "═" * 52 + "╗")
    print(f"║{'  ✅  Portfolio reset to Day-1 state':^52}║")
    print("╠" + "═" * 52 + "╣")
    print(f"║{'  Seed capital   : $10,000.00':^52}║")
    print(f"║{'  TQQQ shares    : 0':^52}║")
    print(f"║{'  Cash           : $10,000.00':^52}║")
    print(f"║{'  Trades YTD     : 0':^52}║")
    print(f"║{'  Kill switch    : OFF':^52}║")
    print(f"║{'  Inception date : ' + str(date.today()):^52}║")
    print("╠" + "═" * 52 + "╣")
    print(f"║{'  GitHub Actions will execute the real':^52}║")
    print(f"║{'  first trade on the next trading day.':^52}║")
    print("╚" + "═" * 52 + "╝")
    print()


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
            "  python3 paper_trade.py                    # daily production run\n"
            "  python3 paper_trade.py --demo             # full simulation, no files touched\n"
            "  python3 paper_trade.py --reset-portfolio  # wipe back to clean Day-1 state\n"
            "  python3 paper_trade.py --dry-run          # compute plan only, no state update\n"
            "  python3 paper_trade.py --status           # print current portfolio\n"
            "  python3 paper_trade.py --reset            # clear today's execution flag\n"
        ),
    )
    parser.add_argument("--demo",             action="store_true",
                        help="Full simulation using real prices — writes to temp dir only, "
                             "production files untouched")
    parser.add_argument("--reset-portfolio",  action="store_true",
                        help="Wipe paper_portfolio.json and paper_trades.csv back to "
                             "clean Day-1 state ($10,000 seed, 0 trades)")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Compute plan and log — do NOT update portfolio state")
    parser.add_argument("--status",           action="store_true",
                        help="Print current portfolio state and exit")
    parser.add_argument("--reset",            action="store_true",
                        help="Clear today's execution flag (use after failed run recovery)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.reset_portfolio:
        reset_portfolio()
        return

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
