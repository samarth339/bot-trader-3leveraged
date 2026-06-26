"""
safety_guard.py — Pre-Flight Safety Checks
============================================
ALL 10 guards must pass before any order is submitted.
First failure wins — subsequent guards are not checked.

Guards (in priority order):
  1.  Kill switch file present             → flatten position, then freeze buys
  2.  Shadow mode not completed            → soft block
  3.  Already executed today               → dedup block
  4.  Outside execution window             → time block
  5.  Portfolio drawdown ≥ halt threshold  → activate kill switch + flatten
  6.  Trades YTD ≥ 100                    → BUY block only (warn at 50)
  7.  TQQQ crash day (≤ −7% vs prev close) → BUY block only
  8.  Position size > 105% NLV            → sanity block
  9.  Live VIX ≥ 45 (extreme)             → BUY block only
  10. TQQQ overnight gap-down > 5%        → BUY block only

Three outcomes:
  hard block    — GuardResult(blocked=True): no order today.
  force flatten — guards 1/5 set signal["_force_flatten"] when shares are
                  held; the reconciler computes a full exit. A halt must
                  REDUCE leveraged exposure, never freeze a TQQQ position.
  buy block     — guards 6/7/9/10 append to self.buy_block_reasons; the
                  executor blocks the order only if the computed plan is a
                  BUY. SELL orders always pass — reducing exposure on a bad
                  day is always permitted.

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
        self.signal  = signal                    # may be mutated (guards 1/5)
        self._state  = state_module.load()
        self.buy_block_reasons: list = []        # applied to BUY plans only

    # ── Entry point ────────────────────────────────────────────────────────────
    def run_all_checks(self) -> GuardResult:
        checks = [
            ("shadow_mode",       self._check_shadow_mode),
            ("double_submission", self._check_double_submission),
            ("market_hours",      self._check_market_hours),
            ("kill_switch",       self._check_kill_switch),      # may force flatten
            ("portfolio_dd",      self._check_portfolio_drawdown),  # may force flatten
            ("trade_frequency",   self._check_trade_frequency),  # BUY block only
            ("crash_day",         self._check_crash_day),        # BUY block only
            ("position_sanity",   self._check_position_sanity),
            ("vix_extreme",       self._check_vix_extreme),      # BUY block only
            ("gap_guard",         self._check_gap_guard),        # BUY block only
        ]
        for name, check_fn in checks:
            result = check_fn()
            if result.blocked:
                logger.warning(f"Guard [{name}] BLOCKED: {result.reason}")
                return result
            logger.debug(f"Guard [{name}] ✓")

        if self.buy_block_reasons:
            logger.warning(
                f"BUY orders blocked today ({len(self.buy_block_reasons)} guard(s)): "
                + " | ".join(self.buy_block_reasons)
            )
        logger.info("All safety guards passed ✓ (hard blocks)")
        return GuardResult(blocked=False)

    # ── Helper: flatten when holding, hard-block when flat ────────────────────
    def _flatten_or_block(self, reason: str) -> GuardResult:
        if self.account is not None and self.account.tqqq_shares() > 0:
            logger.warning(f"{reason} — flattening position (full exit)")
            self.signal["_force_flatten"] = True
            self.signal["_force_flatten_reason"] = reason
            return GuardResult(blocked=False)
        return GuardResult(
            blocked=True,
            reason=f"{reason} — no position held, all trading frozen"
        )

    # ── Guard 1: Kill switch ───────────────────────────────────────────────────
    def _check_kill_switch(self) -> GuardResult:
        if kill_switch.is_active():
            reason = kill_switch.read_reason()
            return self._flatten_or_block(f"Kill switch active — {reason}")
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
            return self._flatten_or_block(msg)

        if drawdown >= DD_WARNING_LEVEL:
            logger.warning(
                f"Portfolio drawdown WARNING: {drawdown:.1%} "
                f"(halt threshold: {limit:.1%})"
            )

        logger.info(f"Portfolio drawdown: {drawdown:.1%}  (limit {limit:.1%}) ✓")
        return GuardResult(blocked=False)

    # ── Guard 6: Trade frequency (BUY block only) ──────────────────────────────
    def _check_trade_frequency(self) -> GuardResult:
        """A frequency cap must never stop risk-reducing sells."""
        ytd = self._state.get("total_trades_ytd", 0)
        if ytd >= MAX_TRADES_PER_YEAR:
            self.buy_block_reasons.append(
                f"YTD trade count ({ytd}) ≥ {MAX_TRADES_PER_YEAR} — BUYs blocked "
                "(frequent trading destroys 3x ETF returns via compounding decay)"
            )
        elif ytd >= MAX_TRADES_PER_YEAR // 2:
            logger.warning(
                f"Trade frequency warning: {ytd}/{MAX_TRADES_PER_YEAR} trades this "
                "year — investigate churn before the cap blocks buying"
            )
        else:
            logger.info(f"Trades YTD: {ytd}/{MAX_TRADES_PER_YEAR} ✓")
        return GuardResult(blocked=False)

    # ── Guard 7: Crash day (BUY block only) ────────────────────────────────────
    def _check_crash_day(self) -> GuardResult:
        """
        TQQQ down ≥ daily_stop_loss vs the PREVIOUS CLOSE → block BUYs today.

        Replaces the old last-fill-relative stop, which force-sold whole
        positions at the close and re-bought full size the next session
        (two losing round trips, −20%, in the first week of paper trading).
        Exits are owned by the replayed strategy state (exposure_a/b in the
        signal) — it sees today's bar tomorrow, exactly like the backtest's
        same-bar stop with one bar of execution lag.
        """
        try:
            data = yf.download("TQQQ", period="5d", interval="1d", progress=False)
            if data.empty or len(data) < 2:
                logger.warning("Crash-day check skipped: insufficient TQQQ history")
                return GuardResult(blocked=False)

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            prev_close = float(data["Close"].iloc[-2])
            last_price = float(data["Close"].iloc[-1])
            if prev_close <= 0:
                return GuardResult(blocked=False)

            day_change = (last_price - prev_close) / prev_close
            limit      = RISK_CONFIG["daily_stop_loss"]
            logger.info(
                f"Crash-day check: TQQQ {day_change:+.1%} vs prev close "
                f"(${prev_close:.2f} → ${last_price:.2f}, buy-block at {-limit:.0%})"
            )
            if day_change <= -limit:
                self.buy_block_reasons.append(
                    f"TQQQ {day_change:+.1%} today (≥ {limit:.0%} drop) — "
                    "no BUYs into a crash; strategy state re-evaluates tomorrow"
                )
        except Exception as exc:
            logger.warning(f"Crash-day check skipped (fetch error): {exc}")

        return GuardResult(blocked=False)   # never hard-blocks

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
                # Only blocks BUY orders — selling is always permitted.
                # Applied by the executor against the computed plan direction.
                self.buy_block_reasons.append(
                    f"Live VIX={live_vix:.1f} ≥ {VIX_EXTREME} (extreme crash) — "
                    "BUY orders blocked as live-market safety override"
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

        if gap.triggered:
            # Applied by the executor against the computed plan direction —
            # SELL orders always pass through.
            self.buy_block_reasons.append(gap.reason)

        return GuardResult(blocked=False)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _compute_blended_target_pct(self) -> float:
        """
        Blended TQQQ allocation for the sanity check. Delegates to the
        reconciler's logic (exposure state when present, max-position caps
        as fallback) so the two paths can never disagree.
        """
        from .position_reconciler import PositionReconciler
        return PositionReconciler.compute_blended_target_pct(self.signal)
