#!/usr/bin/env bash
# phase4_dryrun.sh — Full executor pipeline without submitting orders.
# Connects to IB Gateway, fetches account state, computes rebalance plan,
# logs what would happen — but does NOT place any order.
# Run after phase4_preflight.sh passes.
# Usage: bash scripts/phase4_dryrun.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
LOGFILE="logs/phase4_dryrun.log"

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOGFILE"; }

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 4 Dry-Run  (no orders will be placed)${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════${RESET}"
echo ""
log "Dry-run started"

# ── Check IB Gateway is reachable ─────────────────────────────────────────────
if ! python3 - <<'PYEOF' 2>/dev/null
import socket, sys
try:
    s = socket.create_connection(("127.0.0.1", 4002), timeout=3)
    s.close()
except Exception:
    sys.exit(1)
PYEOF
then
  echo -e "${YELLOW}IB Gateway not detected on port 4002.${RESET}"
  echo "Start IB Gateway (paper account) and re-run."
  exit 1
fi

# ── Pull latest from GitHub ───────────────────────────────────────────────────
log "Pulling latest signal from GitHub..."
git pull --ff-only origin main 2>&1 | tee -a "$LOGFILE"

# ── Refresh market data ───────────────────────────────────────────────────────
log "Refreshing market data (yfinance)..."
python3 data/fetch_data.py 2>&1 | tee -a "$LOGFILE"

# ── Generate today's signal ───────────────────────────────────────────────────
# The executor's read_signal() requires as_of_date == today.
# The GitHub Actions shadow workflow runs at 5:30 PM ET — too late for 3:50 PM execution.
# We generate the signal locally here so the executor always has today's row.
log "Generating today's signal (T-1)..."
python3 daily_signal.py 2>&1 | tee -a "$LOGFILE"

# ── Run executor in dry-run mode ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}── Executor output ─────────────────────────────${RESET}"
echo ""
log "Running IBKR executor --paper --dry-run..."
python3 -m ibkr.executor --paper --dry-run 2>&1 | tee -a "$LOGFILE"

echo ""
log "Dry-run complete."
echo ""
echo -e "  Full log: ${BOLD}logs/phase4_dryrun.log${RESET}"
echo -e "  Execution log: ${BOLD}logs/ibkr_execution.log${RESET}"
echo ""
echo -e "  ${GREEN}If the plan looks correct → bash scripts/phase4_run.sh${RESET}"
echo -e "  ${GREEN}To schedule automatically → bash scripts/phase4_setup_launchd.sh${RESET}"
echo ""
