"""
account.py — Account Summary and Position Fetcher
==================================================
Thin wrapper over ib_insync account/position APIs.

Fetches:
  - Net Liquidation Value (NLV)
  - Available funds (buying power)
  - Cash balance
  - All open positions (ticker → shares, avg cost)

Results are stored in AccountState and cached until refresh() is called again.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from ib_insync import IB

logger = logging.getLogger("ibkr.account")


@dataclass
class AccountState:
    """Snapshot of account value and positions at time of refresh()."""
    net_liquidation:      float = 0.0
    available_funds:      float = 0.0
    cash_balance:         float = 0.0
    gross_position_value: float = 0.0

    # ticker symbol → share count (fractional not supported for leveraged ETFs)
    positions: Dict[str, float] = field(default_factory=dict)
    # ticker symbol → average cost per share
    avg_costs: Dict[str, float] = field(default_factory=dict)

    def tqqq_shares(self) -> int:
        return int(self.positions.get("TQQQ", 0))

    def tqqq_avg_cost(self) -> float:
        return self.avg_costs.get("TQQQ", 0.0)

    def tqqq_market_value(self) -> float:
        """Returns market value based on avg cost (use only as fallback)."""
        return self.tqqq_shares() * self.tqqq_avg_cost()


class AccountManager:
    """
    Fetches and caches IBKR account data.

    Usage:
        mgr     = AccountManager(ib)
        account = mgr.refresh()          # fetch fresh data
        nlv     = account.net_liquidation
        shares  = account.tqqq_shares()
    """

    def __init__(self, ib: IB):
        self.ib     = ib
        self._state: Optional[AccountState] = None

    def refresh(self) -> AccountState:
        """
        Fetch fresh account data from IBKR.
        Populates both account summary values and all open positions.
        """
        state = AccountState()

        # ── Account summary ────────────────────────────────────────────────
        summary = self.ib.accountSummary()
        if not summary:
            raise RuntimeError(
                "accountSummary() returned empty — is the Gateway connected "
                "and the account subscribed?"
            )

        tag_map = {av.tag: av.value for av in summary}

        state.net_liquidation      = float(tag_map.get("NetLiquidation",    0))
        state.available_funds      = float(tag_map.get("AvailableFunds",    0))
        state.cash_balance         = float(tag_map.get("CashBalance",       0))
        state.gross_position_value = float(tag_map.get("GrossPositionValue", 0))

        logger.info(
            f"Account snapshot: "
            f"NLV=${state.net_liquidation:,.2f}  "
            f"Cash=${state.cash_balance:,.2f}  "
            f"Available=${state.available_funds:,.2f}  "
            f"Positions=${state.gross_position_value:,.2f}"
        )

        # ── Positions ──────────────────────────────────────────────────────
        positions = self.ib.positions()
        for pos in positions:
            ticker   = pos.contract.symbol
            shares   = float(pos.position)
            avg_cost = float(pos.avgCost)

            state.positions[ticker] = shares
            state.avg_costs[ticker] = avg_cost

            logger.info(
                f"Position: {ticker:6s}  {shares:>8.0f} shares  "
                f"avg_cost=${avg_cost:.2f}  "
                f"mkt_value=${shares * avg_cost:,.2f}"
            )

        if not state.positions:
            logger.info("No open positions — account is fully in cash")

        self._state = state
        return state

    @property
    def state(self) -> AccountState:
        """Return cached state. Call refresh() first."""
        if self._state is None:
            raise RuntimeError("Call refresh() before accessing account state")
        return self._state
