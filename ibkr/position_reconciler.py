"""
position_reconciler.py — Portfolio Reconciliation (portfolio.py)
=================================================================
Reads:
  - Current IBKR positions (AccountState)
  - Target allocation from the latest row in signal_history.csv
  - Strategy configs (max_position_pct per strategy)

Outputs:
  - RebalancePlan: fully describes what to buy/sell and why

Target TQQQ allocation is computed as a weighted blend of per-strategy caps:

  bull example (weight_a=0.75, weight_b=0.25):
    0.75 × 0.90 (Strategy A cap) = 0.675
    0.25 × 0.70 (Strategy B cap) = 0.175
    blended target               = 0.850  (85% of NLV in TQQQ)

Stagger exit:
  On the first REDUCE_A signal day, exit only to the midpoint between
  current and full target allocation (matching the backtest stagger_exit=True
  behaviour). On the second consecutive REDUCE_A day, go to full target.

Drift gate:
  If actual allocation is within RISK_CONFIG["alloc_drift_rebalance"] (5%)
  of target, no order is submitted — avoids unnecessary churn.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, RISK_CONFIG
from .account import AccountState

logger = logging.getLogger("ibkr.reconciler")

SIGNAL_LOG   = Path("logs/signal_history.csv")
TQQQ_TICKER  = "TQQQ"


@dataclass
class RebalancePlan:
    """
    Fully describes a rebalancing action.

    delta_shares > 0  → BUY
    delta_shares < 0  → SELL
    delta_shares = 0  → no action
    proceed = False   → within drift tolerance or zero delta — skip order
    """
    delta_shares:   int
    target_shares:  int
    current_shares: int
    target_pct:     float     # fraction of NLV (0.0–1.0)
    current_pct:    float
    drift_pct:      float     # |target_pct - current_pct|
    tqqq_price:     float     # reference price used for sizing
    regime:         str
    action:         str
    reason:         str
    proceed:        bool

    def direction(self) -> str:
        if self.delta_shares > 0:
            return "BUY"
        if self.delta_shares < 0:
            return "SELL"
        return "NONE"


class PositionReconciler:
    """
    Compute how the portfolio should be rebalanced given current state + signal.

    Usage:
        reconciler = PositionReconciler(account_state)
        signal     = reconciler.read_signal()
        plan       = reconciler.compute_plan(signal, stagger_day=0)
        adjusted   = reconciler.check_buying_power(plan)
    """

    def __init__(self, account_state: Optional[AccountState]):
        self.account = account_state

    # ── Signal reading ─────────────────────────────────────────────────────────
    def read_signal(self) -> dict:
        """
        Read and validate the most recent row from signal_history.csv.

        Raises:
            FileNotFoundError: signal file does not exist
            ValueError:        file is empty, or signal is stale (not today's)
        """
        if not SIGNAL_LOG.exists():
            raise FileNotFoundError(
                f"Signal log not found: {SIGNAL_LOG}. "
                "Run: python3 daily_signal.py"
            )

        df = pd.read_csv(SIGNAL_LOG, parse_dates=["as_of_date", "signal_date"])
        if df.empty:
            raise ValueError(
                f"Signal log {SIGNAL_LOG} is empty — run daily_signal.py first"
            )

        row         = df.iloc[-1].to_dict()
        signal_date = pd.Timestamp(row["as_of_date"]).date()
        today       = date.today()

        if signal_date != today:
            raise ValueError(
                f"Signal is stale: as_of_date={signal_date}, today={today}. "
                "Run daily_signal.py to generate today's signal before executing."
            )

        logger.info(
            f"Signal loaded: {signal_date}  regime={row['regime']}  "
            f"action={row['action']}  weight_a={row['weight_a']}  "
            f"weight_b={row['weight_b']}  rebalance={row['rebalance']}"
        )
        return row

    # ── Live price ─────────────────────────────────────────────────────────────
    def get_live_tqqq_price(self) -> float:
        """
        Fetch the most recent TQQQ price intraday.
        Falls back to last recorded fill price if yfinance fails.
        """
        try:
            data = yf.download(TQQQ_TICKER, period="1d", interval="1m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.droplevel(1, axis=1)
                price = float(data["Close"].iloc[-1])
                logger.info(f"Live TQQQ price: ${price:.2f}")
                return price
        except Exception as exc:
            logger.warning(f"yfinance TQQQ fetch failed: {exc}")

        # Fallback to last known fill price from state
        from . import state as state_module
        fallback = state_module.load().get("last_fill_price", 0.0)
        if fallback > 0:
            logger.warning(
                f"Using last fill price as TQQQ reference: ${fallback:.2f}"
            )
            return fallback

        raise RuntimeError(
            "Cannot determine TQQQ price — yfinance failed and no fill history. "
            "Aborting to avoid position sizing errors."
        )

    # ── Target allocation ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_exposure(signal: dict, key: str):
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

    @staticmethod
    def compute_blended_target_pct(signal: dict) -> float:
        """
        Compute blended TQQQ target as fraction of NLV.

        Overlays applied (in order):
          1. Force-flatten override — kill switch / DD halt forces 0.0 (full exit).
          2. Exposure state — weight_a×exposure_a + weight_b×exposure_b, where
             exposure_a/b are the replayed per-strategy states written by
             daily_signal.py (backtester/exposure_replay). This equals the
             allocation the validated backtest holds, INCLUDING its MA/VIX
             exits, vol scaling, and crash brake.
          3. Fallback — weight×max_position_pct when exposure fields are
             missing. WARNING: this floors TQQQ exposure at ~66% of NLV even
             in high_vol (the strategies' own de-risking is not applied) and
             produced 60%+ drawdowns in replay. It exists only so a missing
             column cannot crash execution.
        """
        if signal.get("_force_flatten"):
            logger.warning("Force-flatten override: target allocation = 0.0% "
                           f"({signal.get('_force_flatten_reason', 'risk halt')})")
            return 0.0

        weight_a = float(signal.get("weight_a", 0.75))
        weight_b = float(signal.get("weight_b", 0.25))

        exp_a = PositionReconciler._parse_exposure(signal, "exposure_a")
        exp_b = PositionReconciler._parse_exposure(signal, "exposure_b")

        if exp_a is not None and exp_b is not None:
            blended = weight_a * exp_a + weight_b * exp_b
            logger.info(
                f"Blended target (exposure state): "
                f"{weight_a:.0%}×{exp_a:.1%} + {weight_b:.0%}×{exp_b:.1%} "
                f"= {blended:.1%} of NLV in TQQQ"
            )
            return blended

        max_pos_a = STRATEGY_A_CONFIG["max_position_pct"]
        max_pos_b = STRATEGY_B_CONFIG["max_position_pct"]
        blended   = (weight_a * max_pos_a) + (weight_b * max_pos_b)
        logger.error(
            f"Signal row has no exposure_a/exposure_b — falling back to "
            f"max-position caps: {weight_a:.0%}×{max_pos_a:.0%} + "
            f"{weight_b:.0%}×{max_pos_b:.0%} = {blended:.1%} of NLV. "
            "This OVER-ALLOCATES whenever the strategies are de-risked; "
            "re-run daily_signal.py."
        )
        return blended

    # ── Main computation ───────────────────────────────────────────────────────
    def compute_plan(self, signal: dict, stagger_day: int = 0) -> RebalancePlan:
        """
        Compute the rebalancing plan.

        Args:
            signal:      dict from read_signal()
            stagger_day: 0=normal, 1=first REDUCE bar (go halfway),
                         2+=full exit to target

        Returns:
            RebalancePlan describing what to do and why
        """
        if self.account is None:
            raise RuntimeError("account_state is None — pass AccountState to constructor")

        nlv            = self.account.net_liquidation
        current_shares = int(self.account.positions.get(TQQQ_TICKER, 0))
        tqqq_price     = self.get_live_tqqq_price()
        regime         = str(signal.get("regime", "uncertain"))
        action         = str(signal.get("action", "HOLD"))

        # ── Step 1: Target allocation ──────────────────────────────────────────
        target_pct = self.compute_blended_target_pct(signal)

        # ── Step 2: Stagger exit on first REDUCE_A day ────────────────────────
        if stagger_day == 1 and action == "REDUCE_A":
            current_pct   = (current_shares * tqqq_price) / nlv if nlv > 0 else 0
            midpoint_pct  = (current_pct + target_pct) / 2
            logger.info(
                f"Stagger exit (day 1): {current_pct:.1%} → midpoint {midpoint_pct:.1%} "
                f"(full target: {target_pct:.1%})"
            )
            target_pct = midpoint_pct

        # ── Step 3: Compute share counts ──────────────────────────────────────
        target_value  = nlv * target_pct
        target_shares = math.floor(target_value / tqqq_price) if tqqq_price > 0 else 0
        current_pct   = (current_shares * tqqq_price) / nlv if nlv > 0 else 0.0
        drift_pct     = abs(target_pct - current_pct)
        delta_shares  = target_shares - current_shares

        logger.info(
            f"Reconciliation: "
            f"current={current_shares} sh ({current_pct:.1%})  "
            f"target={target_shares} sh ({target_pct:.1%})  "
            f"drift={drift_pct:.1%}  delta={delta_shares:+d}"
        )

        # ── Step 4: Drift gate ─────────────────────────────────────────────────
        # A force-flatten (kill switch / DD halt) always sells to zero — never
        # leave residual leveraged shares behind a halt.
        drift_threshold  = RISK_CONFIG["alloc_drift_rebalance"]
        force_flatten    = bool(signal.get("_force_flatten")) and current_shares > 0
        if force_flatten:
            target_shares, delta_shares = 0, -current_shares
        within_tolerance = drift_pct < drift_threshold and not force_flatten
        proceed = not within_tolerance and delta_shares != 0

        if within_tolerance:
            reason = (
                f"Within drift tolerance ({drift_pct:.1%} < {drift_threshold:.1%}) "
                f"— no order needed"
            )
        elif delta_shares == 0:
            reason = "Delta is zero — already at target share count"
        else:
            direction = "BUY" if delta_shares > 0 else "SELL"
            reason = (
                f"{direction} {abs(delta_shares)} TQQQ  "
                f"({current_pct:.1%} → {target_pct:.1%}  NLV=${nlv:,.0f})"
            )

        return RebalancePlan(
            delta_shares=delta_shares,
            target_shares=target_shares,
            current_shares=current_shares,
            target_pct=target_pct,
            current_pct=current_pct,
            drift_pct=drift_pct,
            tqqq_price=tqqq_price,
            regime=regime,
            action=action,
            reason=reason,
            proceed=proceed,
        )

    # ── Buying power check ─────────────────────────────────────────────────────
    def check_buying_power(self, plan: RebalancePlan) -> int:
        """
        For BUY orders only: verify available funds and reduce delta_shares
        if buying power is insufficient.

        Applies a 5% cost buffer to account for price movement between
        sizing and actual fill.

        Returns:
            Adjusted delta_shares (0 if cannot afford any shares)
        """
        if plan.delta_shares <= 0:
            return plan.delta_shares   # SELL orders: no buying power check needed

        cost_estimate    = plan.delta_shares * plan.tqqq_price * 1.05
        available        = self.account.available_funds

        if cost_estimate <= available:
            logger.info(
                f"Buying power OK: need ${cost_estimate:,.0f}  "
                f"available=${available:,.0f}"
            )
            return plan.delta_shares

        # Reduce to what we can afford
        affordable = math.floor(available / (plan.tqqq_price * 1.05))
        logger.warning(
            f"Buying power limited: wanted {plan.delta_shares} shares "
            f"(${cost_estimate:,.0f}), can afford {affordable} shares "
            f"(${available:,.0f} available) — partial rebalance"
        )
        return max(0, affordable)
