#!/usr/bin/env bash
# phase4_diagnose.sh — Minimal ib_insync connection test.
# Connects, prints account summary, and exits. No orders, no safety guards.
# Use this to confirm IB Gateway is reachable and account data is accessible
# before running the full dry-run.
# Usage: bash scripts/phase4_diagnose.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

echo ""
echo "══════════════════════════════════════"
echo "  IB Gateway diagnostic (paper/4002)"
echo "══════════════════════════════════════"
echo ""

python3 - <<'PYEOF'
import sys
from ib_insync import IB, util

util.startLoop()
ib = IB()

print("Connecting to 127.0.0.1:4002 (paper)...")
try:
    ib.connect("127.0.0.1", 4002, clientId=99, timeout=10, readonly=False)
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print(f"  serverVersion : {ib.client.serverVersion()}")
print(f"  accounts      : {ib.wrapper.accounts}")

print("\nFetching account summary...")
try:
    summary = ib.accountSummary()
    for row in summary:
        if row.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower"):
            print(f"  {row.tag:<22} {row.value:>14}  {row.currency}")
    print("\n  Account data OK ✓")
except Exception as e:
    print(f"  FAILED to fetch account data: {e}")
    ib.disconnect()
    sys.exit(1)

print("\nFetching positions...")
try:
    positions = ib.positions()
    if positions:
        for p in positions:
            print(f"  {p.contract.symbol:<8}  {p.position:>8.0f} shares  avg ${p.avgCost:.2f}")
    else:
        print("  (no positions — paper account is empty)")
    print("  Positions OK ✓")
except Exception as e:
    print(f"  FAILED to fetch positions: {e}")

ib.disconnect()
print("\nDisconnected cleanly.")
print("\nIf you see account data above → run: bash scripts/phase4_dryrun.sh")
PYEOF
