"""
safety_guard.py — Pre-Flight Safety Checks
============================================
ALL 10 guards must pass before any order is submitted.
First failure wins — subsequent guards are not checked.

Guards (in priority order):
  1.  Kill switch file present             → hard block
  2.  Shadow mode not completed            → soft block
  3.  Already executed today               → dedup block
  4.  Outside execution window             → time block
  5.  Portfolio drawdown ≥ 50%             → halt + activate kill switch
  6.  Trades YTD ≥ 100                    → frequency block
  7.  TQQQ daily loss ≥ 7%               → converts to SELL (does not block)
  8.  Position size > 105% NLV            → sanity block
  9.  Live VIX ≥ 45 (extreme)             → BUY block only
  10. TQQQ overnight gap-down > 5%        → BUY block only

Guard 7 does not block — it mutates the signal dict to force a SELL so the
order_manager can exit the position safely at close.

Guard 10 does not block SELL orders — reducing exposure on a gap-down day
is always permitted.  Rationale: 87% of >5% gap-down days close lower than
the open, making new BUY entries at MOC inadvisable.

Usage:
    guard  = SafetyGuard(account_state=account, signal=signal)
    result = guard.run_all_checks()
    if result.blocked:
        abort(result.reason)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import yfinance as yf

from config.strategy_config import RISK_CONFIG, STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
from .account import AccountState
from .gap_guard import GapGuard
from . import kill_switch
from . import state as state_module

logger = logging.getLogger("ibkr.safety")

EST = pytz.timezone("America/New_York")

# ── Execution window ───────────────────────────────────────────────────────────
EXEC_WINDOW_OPEN  = time(15, 45)   # earliest submission time
MOC_DEADLINE      = time(15, 50)   # last time to submit MOC
EXEC_WINDOW_CLOSE = time(15, 58)   # hard cutoff — no orders after this

# ── Thresholds ─────────────────────────────────────────────────────────────────
VIX_EXTREME         = 45.0    # live VIX above this blocks BUY orders
MAX_TRADES_PER_YEAR = 100     # TQQQ decay rule: never exceed this
DD_WARNING_LEVEL    = 0.30    # log warning (not block) at 30% drawdown

SHADOW_STATE_PATH   = Path("logs/shadow_state.json")


@dataclass
class GuardResult:
    blocked: bool
    reason:  str = ""

    def __bool__(self) -> bool:
        """True = allowed to trade (not blocked)."""
        return not self.blocked


class SafetyGuard:
    """
    Run all pre-flight checks. Returns GuardResult on first failure.

    Args:
        account_state: fresh AccountState from AccountManager.refresh()
        signal:        dict from signal_history.csv (may be mutated by guard 7)
    """

    def __init__(self, account_state: AccountState, signal: dict):
        self.account = account_state
        self.signal  = signal                    # may be mutated (guard 7)
        self._state  = state_module.load()

    # ── Entry point ────────────────────────────────────────────────────────────
    def run_all_checks(self) -> GuardResult:
        checks = [
            ("kill_switch",       self._check_kill_switch),
            ("shadow_mode",       self._check_shadow_mode),
            ("double_submission", self._check_double_submission),
            ("market_hours",      self._check_market_hours),
            ("portfolio_dd",      self._check_portfolio_drawdown),
            ("trade_frequency",   self._check_trade_frequency),
            ("daily_loss",        self._check_daily_loss),       # mutates signal
            ("position_sanity",   self._check_position_sanity),
            ("vix_extreme",       self._check_vix_extreme),
            ("gap_guard",         self._check_gap_guard),        # BUY block only
        ]
        for name, check_fn in checks:
            result = check_fn()
            if result.blocked:
                logger.warning(f"Guard [{name}] BLOCKED: {result.reason}")
                return result
            logger.debug(f"Guard [{name}] ✓")

        logger.info("All 10 safety guards passed ✓")
        return GuardResult(blocked=False)

    # ── Guard 1: Kill switch ───────────────────────────────────────────────────
    def _check_kill_switch(self) -> GuardResult:
        if kill_switch.is_active():
            reason = kill_switch.read_reason()
            return GuardResult(
                blocked=True,
                reason=f"Kill switch active — {reason}"
            )
        return GuardResult(blocked=False)

    # ── Guard 2: Shadow mode ───────────────────────────────────────────────────
    def _check_shadow_mode(self) -> GuardResult:
        # Check shadow_state.json completed flag
        if SHADOW_STATE_PATH.exists():
            with open(SHADOW_STATE_PATH) as f:
                shadow = json.load(f)
            if not shadow.get("completed", False):
                day = shadow.get("day", "?")
                return GuardResult(
                    blocked=True,
                    reason=(
                        f"Shadow mode active (day {day}/30). "
                        "Complete the 30-day observation before live execution. "
                        "Run: python3 shadow_mode.py --status"
                    )
                )

        # Check the signal row itself
        if self.signal.get("shadow", True):
            return GuardResult(
                blocked=True,
                reason=(
                    "Signal row has shadow=True. "
                    "Re-run daily_signal.py in live mode (without --shadow)."
                )
            )

        return GuardResult(blocked=False)

    # ── Guard 3: Double submission ─────────────────────────────────────────────
    def _check_double_submission(self) -> GuardResult:
        if state_module.already_executed_today():
            last_id = self._state.get("last_order_id", "unknown")
            return GuardResult(
                blocked=True,
                reason=(
                    f"Already executed today (order_id={last_id}). "
                    "Check logs/ibkr_orders.csv for status."
                )
            )
        return GuardResult(blocked=False)

    # ── Guard 4: Market hours ──────────────────────────────────────────────────
    def _check_market_hours(self) -> GuardResult:
        now     = datetime.now(EST)
        now_t   = now.time()
        weekday = now.weekday()   # 0=Mon … 6=Sun

        if weekday >= 5:
            return GuardResult(
                blocked=True,
                reason=f"Market closed — {now.strftime('%A')} is a weekend"
            )

        if now_t < EXEC_WINDOW_OPEN:
            return GuardResult(
                blocked=True,
                reason=(
                    f"Too early — execution window opens at "
                    f"{EXEC_WINDOW_OPEN.strftime('%H:%M')} EST "
                    f"(now {now_t.strftime('%H:%M')} EST)"
                )
            )

        if now_t > EXEC_WINDOW_CLOSE:
            return GuardResult(
                blocked=True,
                reason=(
                    f"Too late — execution window closed at "
                    f"{EXEC_WINDOW_CLOSE.strftime('%H:%M')} EST. "
                    "No order submitted — signal carries to next session."
                )
            )

        if now_t > MOC_DEADLINE:
            logger.warning(
                f"Past MOC deadline ({MOC_DEADLINE.strftime('%H:%M')} EST) — "
                "order_manager will fall back to limit-close"
            )

        return GuardResult(blocked=False)

    # ── Guard 5: Portfolio drawdown halt ───────────────────────────────────────
    def _check_portfolio_drawdown(self) -> GuardResult:
        peak = self._state.get("peak_equity", 0.0)
        nlv  = self.account.net_liquidation

        if peak <= 0:
            logger.info("No peak equity on record — skipping drawdown check")
            return GuardResult(blocked=False)

        drawdown = (peak - nlv) / peak
        limit    = RISK_CONFIG["max_drawdown_halt"]

        if drawdown >= limit:
            msg = (
                f"Portfolio drawdown {drawdown:.1%} ≥ halt threshold {limit:.1%} "
                f"(peak=${peak:,.0f}, NLV=${nlv:,.0f}) — kill switch activated"
            )
            kill_switch.activate(msg)
            return GuardResult(blocked=True, reason=msg)

        if drawdown >= DD_WARNING_LEVEL:
            logger.warning(
                f"Portfolio drawdown WARNING: {drawdown:.1%} "
                f"(halt threshold: {limit:.1%})"
            )

        logger.info(f"Portfolio drawdown: {drawdown:.1%}  (limit {limit:.1%}) ✓")
        return GuardResult(blocked=False)

    # ── Guard 6: Trade frequency ───────────────────────────────────────────────
    def _check_trade_frequency(self) -> GuardResult:
        ytd = self._state.get("total_trades_ytd", 0)
        if ytd >= MAX_TRADES_PER_YEAR:
            return GuardResult(
                blocked=True,
                reason=(
                    f"YTD trade count ({ytd}) ≥ {MAX_TRADES_PER_YEAR}. "
                    "Frequent trading destroys 3x ETF returns via compounding decay. "
                    "Review strategy — this should never trigger under normal config."
                )
            )
        logger.info(f"Trades YTD: {ytd}/{MAX_TRADES_PER_YEAR} ✓")
        return GuardResult(blocked=False)

    # ── Guard 7: Daily loss on TQQQ (non-blocking — mutates signal) ───────────
    def _check_daily_loss(self) -> GuardResult:
        last_fill = self._state.get("last_fill_price", 0.0)
        if last_fill <= 0:
            return GuardResult(blocked=False)   # no position history

        current_shares = self.account.tqqq_shares()
        if current_shares <= 0:
            return GuardResult(blocked=False)   # not holding TQQQ

        try:
            data = yf.download("TQQQ", period="1d", interval="1m", progress=False)
            if data.empty:
                logger.warning("Could not fetch live TQQQ price for daily loss check")
                return GuardResult(blocked=False)

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            current_price = float(data["Close"].iloc[-1])
            daily_loss    = (last_fill - current_price) / last_fill
            limit         = RISK_CONFIG["daily_stop_loss"]

            logger.info(
                f"Daily loss check: last_fill=${last_fill:.2f}  "
                f"current=${current_price:.2f}  loss={daily_loss:.1%}  "
                f"limit={limit:.1%}"
            )

            if daily_loss >= limit:
                logger.warning(
                    f"Daily stop-loss triggered ({daily_loss:.1%} ≥ {limit:.1%}) — "
                    "overriding signal to SELL (full exit at close)"
                )
                # Mutate the signal so reconciler computes a full-exit plan
                self.signal["_daily_stop_triggered"] = True

        except Exception as exc:
            logger.warning(f"Daily loss check skipped (fetch error): {exc}")

        return GuardResult(blocked=False)   # never blocks — only mutates

    # ── Guard 8: Position size sanity ─────────────────────────────────────────
    def _check_position_sanity(self) -> GuardResult:
        nlv = self.account.net_liquidation
        if nlv <= 0:
            return GuardResult(
                blocked=True,
                reason=f"Net liquidation value is ${nlv:,.2f} — account error"
            )

        blended = self._compute_blended_target_pct()
        target_value = nlv * blended

        if target_value > nlv * 1.05:
            return GuardResult(
                blocked=True,
                reason=(
                    f"Target position ${target_value:,.0f} > 105% of NLV ${nlv:,.0f} — "
                    "position sizing calculation error"
                )
            )

        logger.info(
            f"Position sanity: target={blended:.1%}  "
            f"value=${target_value:,.0f}  NLV=${nlv:,.0f} ✓"
        )
        return GuardResult(blocked=False)

    # ── Guard 9: VIX extreme event override ───────────────────────────────────
    def _check_vix_extreme(self) -> GuardResult:
        try:
            data = yf.download("^VIX", period="1d", interval="5m", progress=False)
            if data.empty:
                logger.warning("Could not fetch live VIX — skipping extreme check")
                return GuardResult(blocked=False)

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            live_vix = float(data["Close"].iloc[-1])
            logger.info(f"Live VIX: {live_vix:.1f}  (extreme threshold: {VIX_EXTREME})")

            if live_vix >= VIX_EXTREME:
                # Only block BUY orders — selling is always permitted
                weight_a = float(self.signal.get("weight_a", 0))
                action   = self.signal.get("action", "HOLD")

                is_buying = action in ("INCREASE_A", "HOLD") and weight_a > 0 and \
                            not self.signal.get("_daily_stop_triggered", False)

                if is_buying:
                    return GuardResult(
                        blocked=True,
                        reason=(
                            f"Live VIX={live_vix:.1f} ≥ {VIX_EXTREME} (extreme crash). "
                            "BUY orders blocked as live-market safety override. "
                            "SELL orders are still permitted."
                        )
                    )
                else:
                    logger.warning(
                        f"VIX extreme ({live_vix:.1f}) but action is SELL/exit — allowing"
                    )

        except Exception as exc:
            logger.warning(f"VIX extreme check skipped (fetch error): {exc}")

        return GuardResult(blocked=False)

    # ── Guard 10: Overnight gap-down protection ────────────────────────────────
    def _check_gap_guard(self) -> GuardResult:
        """
        Block BUY orders when TQQQ has gapped down >5% at the open.

        Sells are always permitted — reducing risk on a bad open is fine.
        The guard is skipped gracefully if price data is unavailable
        (price fetch errors should never block a legitimate execution).
        """
        gap = GapGuard().check()

        if not gap.triggered:
            return GuardResult(blocked=False)

        # Only block if we'd be increasing or maintaining long TQQQ exposure.
        # Same pattern as Guard 9 (vix_extreme) — sells always pass through.
        action   = self.signal.get("action", "HOLD")
        weight_a = float(self.signal.get("weight_a", 0))
        is_buying = (
            action in ("INCREASE_A", "HOLD")
            and weight_a > 0
            and not self.signal.get("_daily_stop_triggered", False)
        )

        if is_buying:
            return GuardResult(blocked=True, reason=gap.reason)

        logger.warning(
            f"Gap guard triggered ({gap.gap_pct*100:+.1f}%) but "
            f"action={action} — SELL orders proceed"
        )
        return GuardResult(blocked=False)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _compute_blended_target_pct(self) -> float:
        """Blended TQQQ allocation = Σ(weight_i × max_position_i)."""
        weight_a  = float(self.signal.get("weight_a", 0.75))
        weight_b  = float(self.signal.get("weight_b", 0.25))
        max_pos_a = STRATEGY_A_CONFIG["max_position_pct"]
        max_pos_b = STRATEGY_B_CONFIG["max_position_pct"]
        return weight_a * max_pos_a + weight_b * max_pos_b
