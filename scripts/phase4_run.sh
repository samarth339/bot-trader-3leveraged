#!/usr/bin/env bash
# phase4_run.sh — Daily paper trading run.
# Called by launchd (see phase4_setup_launchd.sh) at 3:43 PM ET Mon–Fri.
# Flow: git pull → refresh data → generate signal → submit MOC to IBKR paper account.
# Usage: bash scripts/phase4_run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

LOGFILE="logs/phase4_run.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOGFILE"; }
die() { log "FATAL: $*"; exit 1; }

log "══════ Phase 4 paper run start ══════"

# ── Kill switch ────────────────────────────────────────────────────────────────
if [ -f "logs/ibkr_kill.flag" ]; then
  die "Kill switch is active. See logs/ibkr_kill.flag. Deactivate with: python3 -m ibkr.kill_switch deactivate"
fi

# ── IB Gateway reachability ────────────────────────────────────────────────────
if ! python3 - <<'PYEOF' 2>/dev/null
import socket, sys
try:
    s = socket.create_connection(("127.0.0.1", 4002), timeout=3)
    s.close()
except Exception:
    sys.exit(1)
PYEOF
then
  die "IB Gateway not reachable on port 4002. Is it running and logged in?"
fi

# ── Pull latest ────────────────────────────────────────────────────────────────
log "git pull..."
git pull --ff-only origin main >> "$LOGFILE" 2>&1 || log "WARNING: git pull failed — continuing with local data"

# ── Refresh market data ────────────────────────────────────────────────────────
log "Refreshing market data..."
python3 data/fetch_data.py >> "$LOGFILE" 2>&1 || die "Market data refresh failed"

# ── Generate today's signal (T-1) ─────────────────────────────────────────────
# executor.read_signal() requires as_of_date == today.
# GitHub Actions shadow workflow runs at 5:30 PM ET — after our 3:50 PM window.
# We generate the signal here so it's always fresh for execution.
log "Generating today's signal..."
python3 daily_signal.py >> "$LOGFILE" 2>&1 || die "daily_signal.py failed"

# ── Execute ────────────────────────────────────────────────────────────────────
log "Running IBKR executor (paper)..."
python3 -m ibkr.executor --paper 2>&1 | tee -a "$LOGFILE"
RC=${PIPESTATUS[0]}

if [ "$RC" -eq 0 ]; then
  log "══════ Phase 4 paper run complete (success) ══════"
else
  log "══════ Phase 4 paper run complete (exit $RC) ══════"
fi

exit "$RC"
