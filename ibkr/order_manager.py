"""
order_manager.py — Order Construction and Submission (execution.py)
====================================================================
Primary order type: MOC (Market-on-Close)
  - Submitted before 15:50 EST
  - Fills at the closing auction price — closest analog to the backtested
    execution model ("close" price with 10 bps slippage budget)

Fallback order type: Limit-Close (LMT with tif=OPG/MOC)
  - Used if connected after 15:50 EST but before 15:58 EST
  - Limit price = last price ± slippage buffer

All orders are appended to logs/ibkr_orders.csv (append-only, never modified).
Partial fills are tracked in the order record.

Usage:
    mgr    = OrderManager(ib)
    result = mgr.submit_order(plan, dry_run=False)
    if result.status == "error":
        handle_failure(result.error)
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
from ib_insync import IB, Stock, Order, Trade

from config.strategy_config import EXECUTION_CONFIG
from .position_reconciler import RebalancePlan

logger = logging.getLogger("ibkr.orders")

EST          = pytz.timezone("America/New_York")
ORDERS_LOG   = Path("logs/ibkr_orders.csv")
MOC_CUTOFF   = datetime.strptime("15:50", "%H:%M").time()
SLIPPAGE_BPS = EXECUTION_CONFIG["slippage_bps"]   # 10

ORDER_LOG_FIELDS = [
    "date", "time_est", "ticker", "direction", "quantity",
    "order_type", "tif", "limit_price",
    "order_id", "status",
    "fill_price", "fill_qty", "slippage_bps",
    "reference_price", "regime", "target_pct", "current_pct",
    "dry_run", "error",
]


@dataclass
class OrderResult:
    """Result of a single order submission attempt."""
    order_id:     Optional[str]
    status:       str       # submitted | filled | no_action | dry_run | error
    fill_price:   float     # 0.0 until async fill arrives
    fill_qty:     int
    slippage_bps: float     # computed vs reference price (0.0 if no fill yet)
    error:        str = ""


class OrderManager:
    """
    Construct and submit orders to IBKR via ib_insync.

    Responsibilities:
      - Choose MOC vs limit-close based on current time
      - Qualify contracts before submission
      - Wait for order acknowledgement (not fill — fills are async for MOC)
      - Log every order attempt to ibkr_orders.csv regardless of outcome
      - Track slippage vs reference price
    """

    def __init__(self, ib: IB):
        self.ib = ib
        ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Main entry point ───────────────────────────────────────────────────────
    def submit_order(
        self,
        plan:    RebalancePlan,
        dry_run: bool = False,
    ) -> OrderResult:
        """
        Submit a MOC or limit-close order based on the RebalancePlan.

        Args:
            plan:    computed by PositionReconciler.compute_plan()
            dry_run: if True, log the order but do NOT submit to IBKR

        Returns:
            OrderResult with submission status and fill data (async fills
            will have fill_price=0 until the fill callback fires).
        """
        # ── No action needed ──────────────────────────────────────────────────
        if not plan.proceed or plan.delta_shares == 0:
            logger.info(f"No order required — {plan.reason}")
            result = OrderResult(
                order_id=None, status="no_action",
                fill_price=0.0, fill_qty=0, slippage_bps=0.0,
            )
            self._log_order(plan, "NONE", 0, "NONE", "NONE", 0.0,
                            "NO_ACTION", "no_action", 0.0, 0, 0.0, dry_run)
            return result

        direction = plan.direction()           # "BUY" or "SELL"
        quantity  = abs(plan.delta_shares)

        # ── Choose order type ─────────────────────────────────────────────────
        now_est    = datetime.now(EST).time()
        order_type = "MOC" if now_est <= MOC_CUTOFF else "LMT"

        # ── Build contract ────────────────────────────────────────────────────
        contract = Stock(symbol="TQQQ", exchange="SMART", currency="USD")

        # ── Build order ───────────────────────────────────────────────────────
        limit_price = 0.0

        if order_type == "MOC":
            ibkr_order = Order(
                action        = direction,
                orderType     = "MOC",
                totalQuantity = quantity,
                tif           = "DAY",
                outsideRth    = False,
            )
            logger.info(
                f"Order: {direction} {quantity} TQQQ via MOC "
                f"(execution at closing auction)"
            )

        else:
            # Limit-close: add slippage buffer so we participate in the auction
            slippage_factor = SLIPPAGE_BPS / 10_000
            if direction == "BUY":
                limit_price = round(plan.tqqq_price * (1 + slippage_factor), 2)
            else:
                limit_price = round(plan.tqqq_price * (1 - slippage_factor), 2)

            ibkr_order = Order(
                action        = direction,
                orderType     = "LMT",
                totalQuantity = quantity,
                lmtPrice      = limit_price,
                tif           = "MOC",    # closing auction only
                outsideRth    = False,
            )
            logger.info(
                f"Order: {direction} {quantity} TQQQ via LMT @ ${limit_price:.2f} "
                f"tif=MOC (past MOC cutoff — using limit-close fallback)"
            )

        # ── Dry run ───────────────────────────────────────────────────────────
        if dry_run:
            logger.info(
                f"[DRY RUN] Would submit: {direction} {quantity} TQQQ "
                f"via {order_type}  ref=${plan.tqqq_price:.2f}"
            )
            self._log_order(
                plan, direction, quantity, order_type, "DAY", limit_price,
                "DRY_RUN", "dry_run", 0.0, 0, 0.0, dry_run=True
            )
            return OrderResult(
                order_id="DRY_RUN", status="dry_run",
                fill_price=0.0, fill_qty=0, slippage_bps=0.0,
            )

        # ── Submit ────────────────────────────────────────────────────────────
        try:
            self.ib.qualifyContracts(contract)
            trade: Trade = self.ib.placeOrder(contract, ibkr_order)
            order_id     = str(trade.order.orderId)

            logger.info(
                f"Order submitted ✓  id={order_id}  "
                f"{direction} {quantity} TQQQ via {order_type}"
            )

            # Wait for IB to acknowledge (PreSubmitted → Submitted)
            # MOC fills happen at 16:00 EST — we don't block for the fill here
            ack_timeout_secs = 15
            for _ in range(ack_timeout_secs * 10):
                self.ib.sleep(0.1)
                status = trade.orderStatus.status
                if status not in ("", "PreSubmitted", "PendingSubmit"):
                    break

            status    = trade.orderStatus.status or "Submitted"
            fill_px   = float(trade.orderStatus.avgFillPrice or 0.0)
            fill_qty  = int(trade.orderStatus.filled         or 0)

            # Slippage vs reference price (only meaningful on immediate fills)
            slippage = 0.0
            if fill_px > 0 and plan.tqqq_price > 0:
                if direction == "BUY":
                    slippage = (fill_px - plan.tqqq_price) / plan.tqqq_price * 10_000
                else:
                    slippage = (plan.tqqq_price - fill_px) / plan.tqqq_price * 10_000

            logger.info(
                f"Order ack: id={order_id}  status={status}  "
                f"filled={fill_qty}  avg_fill=${fill_px:.2f}  "
                f"slippage={slippage:.1f} bps"
            )

            self._log_order(
                plan, direction, quantity, order_type, "DAY", limit_price,
                order_id, status, fill_px, fill_qty, slippage, dry_run=False
            )

            return OrderResult(
                order_id=order_id, status=status,
                fill_price=fill_px, fill_qty=fill_qty, slippage_bps=slippage,
            )

        except Exception as exc:
            logger.error(f"Order submission FAILED: {exc}")
            self._log_order(
                plan, direction, quantity, order_type, "DAY",
                limit_price if limit_price else 0.0,
                "FAILED", "error", 0.0, 0, 0.0,
                dry_run=False, error=str(exc)
            )
            return OrderResult(
                order_id=None, status="error",
                fill_price=0.0, fill_qty=0, slippage_bps=0.0,
                error=str(exc),
            )

    # ── Logging ────────────────────────────────────────────────────────────────
    def _log_order(
        self,
        plan:        RebalancePlan,
        direction:   str,
        quantity:    int,
        order_type:  str,
        tif:         str,
        limit_price: float,
        order_id:    str,
        status:      str,
        fill_price:  float,
        fill_qty:    int,
        slippage:    float,
        dry_run:     bool,
        error:       str = "",
    ):
        """Append one row to ibkr_orders.csv — never overwrites existing rows."""
        now           = datetime.now(EST)
        write_header  = not ORDERS_LOG.exists()

        with open(ORDERS_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ORDER_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "date":            now.strftime("%Y-%m-%d"),
                "time_est":        now.strftime("%H:%M:%S"),
                "ticker":          "TQQQ",
                "direction":       direction,
                "quantity":        quantity,
                "order_type":      order_type,
                "tif":             tif,
                "limit_price":     round(limit_price, 4),
                "order_id":        order_id,
                "status":          status,
                "fill_price":      round(fill_price, 4),
                "fill_qty":        fill_qty,
                "slippage_bps":    round(slippage, 2),
                "reference_price": round(plan.tqqq_price, 4),
                "regime":          plan.regime,
                "target_pct":      round(plan.target_pct, 4),
                "current_pct":     round(plan.current_pct, 4),
                "dry_run":         dry_run,
                "error":           error,
            })
        logger.debug(f"Order logged → {ORDERS_LOG}  order_id={order_id}")
