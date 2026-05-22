"""
executor.py — Main Orchestrator  (main.py)
==========================================
Coordinates all IBKR modules into a single execution flow.

Execution sequence (runs daily at 15:50 EST):
  1.  Read today's signal from logs/signal_history.csv
  2.  Connect to IB Gateway (with exponential backoff)
  3.  Fetch account state (NLV, positions, buying power)
  4.  Run all 9 safety guards
  5.  Compute rebalancing plan (target allocation → delta shares)
  6.  Adjust for buying power limits (BUY orders only)
  7.  Submit MOC order (or limit-close if past 15:50)
  8.  Update persistent state (ibkr_state.json)
  9.  Log execution summary
  10. Send email alert
  11. Disconnect cleanly

Usage:
  python -m ibkr.executor                # paper account (default)
  python -m ibkr.executor --live         # live account  ⚠️
  python -m ibkr.executor --dry-run      # connect + compute but do NOT submit
  python -m ibkr.executor --status       # print current state and exit
  python -m ibkr.executor --reset-today  # clear today's execution flag (recovery)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytz

# ── Logging (set up before any module imports so all loggers inherit) ──────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

EST = pytz.timezone("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "ibkr_execution.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ibkr.executor")


def run(paper: bool = True, dry_run: bool = False) -> bool:
    """
    Full execution flow. Returns True on success / clean no-action,
    False if blocked, failed, or errored.

    Args:
        paper:   True = IB Gateway paper port (4002), False = live port (4001)
        dry_run: True = compute plan and log, but do not submit order
    """
    from .client              import IBClient, ConnectionConfig, GATEWAY_PAPER_PORT, GATEWAY_LIVE_PORT
    from .account             import AccountManager
    from .safety_guard        import SafetyGuard
    from .position_reconciler import PositionReconciler
    from .order_manager       import OrderManager
    from .                    import state as state_module
    from .                    import kill_switch

    env  = "PAPER" if paper else "LIVE ⚠️"
    mode = "[DRY RUN]" if dry_run else ""
    now  = datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S %Z")

    logger.info("=" * 65)
    logger.info(f"  IBKR Executor  |  {env}  {mode}  |  {now}")
    logger.info("=" * 65)

    # ── Step 1: Read signal ────────────────────────────────────────────────────
    try:
        reconciler_tmp = PositionReconciler(account_state=None)
        signal = reconciler_tmp.read_signal()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Signal read failed: {exc}")
        _send_alert(
            subject=f"[{env}] IBKR BLOCKED — signal error",
            body=f"Could not read today's signal:\n{exc}\n\n"
                 f"Run: python3 daily_signal.py",
        )
        return False

    # ── Step 2: Connect to IB Gateway ─────────────────────────────────────────
    port   = GATEWAY_PAPER_PORT if paper else GATEWAY_LIVE_PORT
    config = ConnectionConfig(
        host      = "127.0.0.1",
        port      = port,
        client_id = 11 if paper else 10,
        paper     = paper,
        timeout   = 20.0,
    )

    # Hard deadline: give up connecting if past 15:58 EST
    connection_deadline = 5 * 60   # 5 minutes

    with IBClient(config) as client:
        if not client.is_connected():
            logger.error("IB Gateway connection failed after all retries")
            _send_alert(
                subject=f"[{env}] IBKR FAILED — Gateway unreachable",
                body=(
                    f"Could not connect to IB Gateway at {config.host}:{port}.\n\n"
                    "Check that IB Gateway is running and auto-restart is configured.\n"
                    "No order was submitted."
                ),
            )
            return False

        ib = client.ib

        # ── Step 3: Fetch account state ────────────────────────────────────────
        try:
            account_mgr = AccountManager(ib)
            account     = account_mgr.refresh()
        except Exception as exc:
            logger.error(f"Account fetch failed: {exc}")
            _send_alert(
                subject=f"[{env}] IBKR FAILED — account fetch error",
                body=f"accountSummary() or positions() failed:\n{exc}",
            )
            return False

        # ── Step 4: Safety guards ──────────────────────────────────────────────
        guard  = SafetyGuard(account_state=account, signal=signal)
        result = guard.run_all_checks()

        if result.blocked:
            logger.warning(f"Execution BLOCKED: {result.reason}")
            _send_alert(
                subject=f"[{env}] IBKR BLOCKED — {result.reason[:50]}",
                body=(
                    f"Execution blocked by safety guard:\n\n"
                    f"{result.reason}\n\n"
                    f"Account NLV: ${account.net_liquidation:,.2f}\n"
                    f"Signal: {signal.get('regime', '?')} / {signal.get('action', '?')}"
                ),
            )
            return False

        # ── Step 5: Compute rebalancing plan ───────────────────────────────────
        exec_state  = state_module.load()
        stagger_day = exec_state.get("consecutive_reduce_days", 0)

        reconciler = PositionReconciler(account_state=account)
        plan       = reconciler.compute_plan(signal, stagger_day=stagger_day)

        # ── Step 6: Buying power check (BUY orders only) ───────────────────────
        if plan.delta_shares > 0:
            adjusted = reconciler.check_buying_power(plan)
            if adjusted != plan.delta_shares:
                logger.warning(
                    f"Buying power limit: reducing delta from "
                    f"{plan.delta_shares} → {adjusted} shares"
                )
                plan.delta_shares = adjusted
                plan.proceed      = adjusted > 0

        logger.info(f"Plan: {plan.reason}")

        # ── Step 7: Submit order ───────────────────────────────────────────────
        order_mgr    = OrderManager(ib)
        order_result = order_mgr.submit_order(plan, dry_run=dry_run)

        # ── Step 8: Update persistent state ───────────────────────────────────
        if order_result.status not in ("no_action", "dry_run", "error"):
            # Update stagger exit counter
            if signal.get("action") == "REDUCE_A":
                exec_state["consecutive_reduce_days"] = stagger_day + 1
            else:
                exec_state["consecutive_reduce_days"] = 0
            state_module.save(exec_state)

            # Record execution (fill may be async — use reference price as fallback)
            if not dry_run:
                fill_price_to_record = (
                    order_result.fill_price
                    if order_result.fill_price > 0
                    else plan.tqqq_price
                )
                state_module.record_execution(
                    regime          = plan.regime,
                    target_pct      = plan.target_pct,
                    shares_tqqq     = plan.target_shares,
                    fill_price      = fill_price_to_record,
                    net_liquidation = account.net_liquidation,
                    order_id        = order_result.order_id,
                )
        elif order_result.status == "error":
            logger.error(f"Order error: {order_result.error}")

        # ── Step 9: Log summary ────────────────────────────────────────────────
        _log_summary(plan, order_result, account, signal, env, dry_run)

        # ── Step 10: Email ─────────────────────────────────────────────────────
        _send_alert(
            subject=_build_subject(plan, order_result, env, dry_run),
            body=_build_body(plan, order_result, account, signal, env, dry_run),
        )

    # ── Step 11: Disconnect (handled by context manager __exit__) ─────────────
    logger.info("Executor finished cleanly ✓")
    return order_result.status != "error"


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _log_summary(plan, order_result, account, signal, env, dry_run):
    tag = "[DRY RUN]" if dry_run else ""
    logger.info("-" * 65)
    logger.info(f"  EXECUTION SUMMARY  {tag}")
    logger.info(f"  Regime:        {plan.regime.upper()}")
    logger.info(f"  Action:        {plan.action}")
    logger.info(f"  Current alloc: {plan.current_pct:.1%} TQQQ  "
                f"({plan.current_shares} shares)")
    logger.info(f"  Target alloc:  {plan.target_pct:.1%} TQQQ  "
                f"({plan.target_shares} shares)")
    logger.info(f"  Drift:         {plan.drift_pct:.1%}")
    logger.info(f"  Delta:         {plan.delta_shares:+d} shares")
    logger.info(f"  Ref price:     ${plan.tqqq_price:.2f}")
    logger.info(f"  NLV:           ${account.net_liquidation:,.2f}")
    logger.info(f"  Order status:  {order_result.status}")
    if order_result.fill_price:
        logger.info(f"  Fill price:    ${order_result.fill_price:.2f}")
    if order_result.slippage_bps:
        logger.info(f"  Slippage:      {order_result.slippage_bps:.1f} bps")
    if order_result.error:
        logger.error(f"  Error:         {order_result.error}")
    logger.info("-" * 65)


def _build_subject(plan, order_result, env, dry_run) -> str:
    tag  = "[DRY RUN] " if dry_run else ""
    date = datetime.now(EST).strftime("%Y-%m-%d")
    if order_result.status == "no_action":
        return f"[{env}] {tag}IBKR — HOLD {plan.regime.upper()} {date}"
    elif order_result.status == "error":
        return f"[{env}] {tag}IBKR — ERROR {date}"
    else:
        direction = plan.direction()
        return (
            f"[{env}] {tag}IBKR — {direction} {abs(plan.delta_shares)} TQQQ "
            f"({plan.regime.upper()}) {date}"
        )


def _build_body(plan, order_result, account, signal, env, dry_run) -> str:
    now  = datetime.now(EST).strftime("%Y-%m-%d %H:%M EST")
    tag  = " [DRY RUN — no order submitted]" if dry_run else ""

    lines = [
        f"IBKR Execution Report — {env}{tag}",
        f"Time: {now}",
        "",
        f"Regime:        {plan.regime.upper()}",
        f"Action:        {plan.action}",
        f"QQQ SMA-150:   {'above' if float(signal.get('pct_vs_sma', 0)) >= 0 else 'below'}  "
        f"({signal.get('pct_vs_sma', 'n/a')}%)",
        f"VIX (5d avg):  {signal.get('vix_signal', 'n/a')}",
        "",
        f"Current alloc: {plan.current_pct:.1%} ({plan.current_shares} shares TQQQ)",
        f"Target alloc:  {plan.target_pct:.1%} ({plan.target_shares} shares TQQQ)",
        f"Drift:         {plan.drift_pct:.1%}",
        f"Delta:         {plan.delta_shares:+d} shares",
        "",
        f"Net Liquidation: ${account.net_liquidation:,.2f}",
        f"Available funds: ${account.available_funds:,.2f}",
        f"Cash:            ${account.cash_balance:,.2f}",
        "",
        f"Order status:  {order_result.status}",
        f"Order ID:      {order_result.order_id or 'n/a'}",
    ]

    if order_result.fill_price:
        lines.append(f"Fill price:    ${order_result.fill_price:.2f}")
    if order_result.slippage_bps:
        lines.append(f"Slippage:      {order_result.slippage_bps:.1f} bps")
    if order_result.error:
        lines += ["", f"ERROR: {order_result.error}"]

    lines += [
        "",
        "Logs: logs/ibkr_execution.log",
        "Orders: logs/ibkr_orders.csv",
    ]
    return "\n".join(lines)


def _send_alert(subject: str, body: str):
    try:
        import send_email
        send_email.send_email(subject=subject, body=body)
        logger.debug(f"Alert sent: {subject}")
    except Exception as exc:
        logger.warning(f"Email alert failed (non-critical): {exc}")


# ── Status command ─────────────────────────────────────────────────────────────

def print_status():
    from . import state as state_module
    from . import kill_switch

    state = state_module.load()
    ks    = kill_switch.status()

    print()
    print("═" * 55)
    print("  IBKR Executor — Status")
    print("═" * 55)
    print(f"  Kill switch:        {'ACTIVE ⛔' if ks['active'] else 'OFF ✓'}")
    if ks["reason"]:
        print(f"  Kill reason:        {ks['reason'][:60]}")
    print(f"  Last execution:     {state.get('last_execution_date') or 'never'}")
    print(f"  Last regime:        {state.get('last_regime') or 'n/a'}")
    print(f"  Last target alloc:  {state.get('last_target_pct', 0):.1%}")
    print(f"  Last TQQQ shares:   {state.get('last_shares_tqqq', 0)}")
    print(f"  Last fill price:    ${state.get('last_fill_price', 0):.2f}")
    print(f"  Peak equity:        ${state.get('peak_equity', 0):,.2f}")
    print(f"  Trades YTD:         {state.get('total_trades_ytd', 0)}")
    print(f"  Consec. reduce:     {state.get('consecutive_reduce_days', 0)}")
    print(f"  Last order ID:      {state.get('last_order_id') or 'n/a'}")
    print("═" * 55)
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IBKR Trade Executor — T-1 signal → MOC order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m ibkr.executor --paper --dry-run\n"
            "  python -m ibkr.executor --live\n"
            "  python -m ibkr.executor --status\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True,
                       help="Use IB Gateway paper account on port 4002 (default)")
    group.add_argument("--live",  action="store_true", default=False,
                       help="Use IB Gateway live account on port 4001 ⚠️")

    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Connect, compute plan, log — but do NOT submit order")
    parser.add_argument("--status",  action="store_true", default=False,
                        help="Print current execution state and exit")
    parser.add_argument("--reset-today", action="store_true", default=False,
                        help="Clear today's execution flag (use after failed run recovery)")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.reset_today:
        from . import state as state_module
        s = state_module.load()
        s["last_execution_date"] = None
        state_module.save(s)
        print("Execution flag cleared — ready to re-run today")
        return

    paper   = not args.live
    dry_run = args.dry_run

    if not paper:
        confirm = input(
            "⚠️  LIVE account mode. Real money will be traded. "
            "Type 'yes' to confirm: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    success = run(paper=paper, dry_run=dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
