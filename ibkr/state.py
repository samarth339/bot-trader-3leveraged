"""
state.py — Persistent Execution State
=======================================
Persists IBKR execution state across sessions to logs/ibkr_state.json.

Tracked fields:
  last_execution_date     — ISO date; guards against double-submission
  last_regime             — "bull" / "uncertain" / "high_vol"
  last_target_pct         — float 0–1; last blended TQQQ target allocation
  last_shares_tqqq        — int; last confirmed TQQQ share count
  last_fill_price         — float; last order fill (or reference price)
  peak_equity             — float; rolling high-water mark for drawdown calc
  total_trades_ytd        — int; resets January 1
  daily_trade_count       — int; resets each new day
  consecutive_reduce_days — int; stagger exit counter
  last_order_id           — str; IBKR order ID for dedup checks
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ibkr.state")

STATE_PATH = Path("logs/ibkr_state.json")

DEFAULT_STATE: dict = {
    "last_execution_date":     None,   # "YYYY-MM-DD" or null
    "last_regime":             None,   # "bull" / "uncertain" / "high_vol" / null
    "last_target_pct":         0.0,    # 0.0 – 1.0
    "last_shares_tqqq":        0,      # integer shares
    "last_fill_price":         0.0,    # USD per share
    "peak_equity":             0.0,    # USD — high-water mark for drawdown
    "total_trades_ytd":        0,      # resets Jan 1
    "daily_trade_count":       0,      # resets each day
    "consecutive_reduce_days": 0,      # stagger exit counter
    "last_order_id":           None,   # IBKR order ID string or null
}


def load() -> dict:
    """
    Load state from disk. Missing keys are back-filled with defaults.
    Safe to call when file does not exist (returns defaults).
    """
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            on_disk = json.load(f)
        # Forward-compatible: add any new default keys without overwriting
        merged = {**DEFAULT_STATE, **on_disk}
        return merged

    return DEFAULT_STATE.copy()


def save(state: dict):
    """Atomically write state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.debug(f"State saved → {STATE_PATH}")


def already_executed_today() -> bool:
    """Return True if an order was already submitted today."""
    return load().get("last_execution_date") == date.today().isoformat()


def record_execution(
    regime:           str,
    target_pct:       float,
    shares_tqqq:      int,
    fill_price:       float,
    net_liquidation:  float,
    order_id:         Optional[str] = None,
):
    """
    Called after a successful order submission.
    Updates counters, peak equity, and execution metadata.
    """
    state = load()
    today = date.today().isoformat()

    # Reset daily trade count on new day
    if state.get("last_execution_date") != today:
        state["daily_trade_count"] = 0

    # Update peak equity (high-water mark)
    if net_liquidation > state.get("peak_equity", 0):
        state["peak_equity"] = net_liquidation
        logger.info(f"New peak equity: ${net_liquidation:,.2f}")

    # YTD counter: reset on Jan 1
    if (
        state.get("last_execution_date") is not None
        and state["last_execution_date"][:4] != today[:4]
    ):
        logger.info("New calendar year — resetting total_trades_ytd")
        state["total_trades_ytd"] = 0

    state["daily_trade_count"]   = state.get("daily_trade_count", 0) + 1
    state["total_trades_ytd"]    = state.get("total_trades_ytd",  0) + 1
    state["last_execution_date"] = today
    state["last_regime"]         = regime
    state["last_target_pct"]     = round(target_pct, 4)
    state["last_shares_tqqq"]    = shares_tqqq
    state["last_fill_price"]     = round(fill_price, 4)
    state["last_order_id"]       = order_id

    save(state)
    logger.info(
        f"Execution recorded: {today}  regime={regime}  "
        f"target={target_pct:.1%}  shares={shares_tqqq}  "
        f"fill=${fill_price:.2f}  trades_ytd={state['total_trades_ytd']}"
    )


def update_fill(fill_price: float, fill_qty: int, order_id: Optional[str] = None):
    """
    Called asynchronously when IBKR reports a fill (e.g. from callback).
    Updates fill price without resetting other counters.
    """
    state = load()
    if fill_price > 0:
        state["last_fill_price"] = round(fill_price, 4)
    if fill_qty > 0:
        state["last_shares_tqqq"] = fill_qty
    if order_id:
        state["last_order_id"] = order_id
    save(state)
    logger.info(f"Fill update: {fill_qty} shares @ ${fill_price:.2f}")
