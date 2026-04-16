"""
ibkr/ — Interactive Brokers Live Execution Layer
=================================================
Phase 4/5 module. Reads T-1 signals from signal_history.csv and
executes MOC (Market-on-Close) orders via IB Gateway.

Modules:
  client.py              — IB Gateway connection + reconnect logic
  account.py             — Account summary + position fetcher
  kill_switch.py         — Hard stop (file-based flag)
  state.py               — Persistent execution state (ibkr_state.json)
  safety_guard.py        — 9 pre-flight safety checks
  position_reconciler.py — Compute target allocation + delta shares
  order_manager.py       — MOC/limit-close order submission + logging
  executor.py            — Main orchestrator (entry point)

Usage:
  python -m ibkr.executor --paper        # Paper account (default)
  python -m ibkr.executor --live         # Live account
  python -m ibkr.executor --dry-run      # Log only, no order
  python -m ibkr.executor --status       # Print state summary

Safety gates (ALL must pass before any order):
  1. Kill switch file not present
  2. Shadow mode completed (30-day observation done)
  3. Not already executed today
  4. Within 15:45–15:58 EST execution window
  5. Portfolio drawdown < 50%
  6. Trades YTD < 100
  7. TQQQ daily loss < 7%
  8. Position size sane (< 105% NLV)
  9. Live VIX < 45
"""
