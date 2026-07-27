#!/bin/bash
# ============================================================================
#  Trading Dashboard — one-double-click launcher (macOS)
# ----------------------------------------------------------------------------
#  Double-click this file (or an alias of it on your Desktop) to:
#    1. move into the project,
#    2. verify Python + dependencies,
#    3. pull the latest committed data (if the repo is clean),
#    4. start the dashboard server if it isn't already running,
#    5. open the dashboard in your browser.
#  Errors are shown in plain language and the window stays open so you can read
#  them. Close the window (or Ctrl-C) to stop the server.
# ============================================================================

set -o pipefail
PORT=8050
URL="http://127.0.0.1:${PORT}"
GREEN='\033[0;32m'; RED='\033[0;31m'; AMBER='\033[0;33m'; DIM='\033[2m'; NC='\033[0m'

# Always run from the project directory (folder containing this script).
cd "$(dirname "$0")" || { echo "Cannot find the project folder."; read -r; exit 1; }
PROJECT="$(pwd)"

banner() { echo -e "\n${GREEN}▌ Trading Dashboard${NC} ${DIM}— $PROJECT${NC}\n"; }
fail()   { echo -e "\n${RED}✗ $1${NC}\n\nPress any key to close…"; read -n 1 -s -r; exit 1; }

banner

# ── 1. Python ────────────────────────────────────────────────────────────────
PY="$(command -v python3)"
[ -z "$PY" ] && fail "python3 is not installed. Install Python 3, then try again."
echo -e "  ${GREEN}✓${NC} python3  ($("$PY" --version 2>&1))"

# ── 2. Dependencies (dash) ───────────────────────────────────────────────────
if ! "$PY" -c "import dash, dash_bootstrap_components, plotly, pandas" 2>/dev/null; then
  echo -e "  ${AMBER}!${NC} Dashboard dependencies missing — installing (one-time)…"
  "$PY" -m pip install -q -r requirements.txt 2>/dev/null \
    || "$PY" -m pip install -q dash dash-bootstrap-components plotly pandas numpy \
    || fail "Could not install dependencies. Run:  python3 -m pip install -r requirements.txt"
fi
echo -e "  ${GREEN}✓${NC} dependencies"

# ── 3. Sync latest committed run data from GitHub (signals, trades, portfolio) ─
# The daily GitHub Actions runs COMMIT signal_history.csv / paper_portfolio.json /
# paper_trades.csv. A fast-forward pull brings all of them in. Untracked files
# (like this launcher's own log) do NOT block --ff-only, so we always try it.
if command -v git >/dev/null && git rev-parse --git-dir >/dev/null 2>&1; then
  git fetch --quiet origin 2>/dev/null
  BEHIND=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)
  if git pull --ff-only --quiet 2>/dev/null; then
    if [ "${BEHIND:-0}" -gt 0 ]; then
      echo -e "  ${GREEN}✓${NC} synced ${BEHIND} new commit(s) from GitHub (past runs)"
    else
      echo -e "  ${GREEN}✓${NC} already up to date with GitHub"
    fi
  else
    echo -e "  ${AMBER}!${NC} could not fast-forward (offline, or local edits to tracked files)."
    echo -e "     ${DIM}Showing local data. To force-sync:  git stash && git pull${NC}"
  fi
fi

# ── 3b. Refresh market prices so charts & unrealized P/L are current ──────────
# Prices (data/processed) are NOT committed by the runs, so refresh them here.
# Non-fatal: offline ⇒ use cached data. Set SKIP_REFRESH=1 to skip for speed.
if [ "${SKIP_REFRESH:-0}" != "1" ]; then
  echo -e "  ${DIM}· refreshing market prices (a few seconds)…${NC}"
  if PYTHONPATH="$PROJECT" "$PY" data/fetch_data.py >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} prices up to date"
  else
    echo -e "  ${DIM}· price refresh skipped (offline) — using cached prices${NC}"
  fi
fi

# ── 4. Start server if not already running ───────────────────────────────────
if lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo -e "  ${GREEN}✓${NC} dashboard already running on port ${PORT}"
else
  echo -e "  ${DIM}· starting server (first paint runs one backtest, ~5s)…${NC}"
  mkdir -p logs
  PYTHONPATH="$PROJECT" nohup "$PY" -m dashboard.app > logs/dashboard.log 2>&1 &
  SERVER_PID=$!
  # wait up to 60s for the port to come up
  for i in $(seq 1 60); do
    if lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo -e "\n${RED}Server exited during startup. Last log lines:${NC}"
      tail -n 15 logs/dashboard.log
      fail "Dashboard failed to start (see logs/dashboard.log)."
    fi
    sleep 1
  done
  lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1 \
    || fail "Dashboard did not come up on port ${PORT} within 60s (see logs/dashboard.log)."
  echo -e "  ${GREEN}✓${NC} server up (pid $SERVER_PID)"
fi

# ── 5. Open the browser ──────────────────────────────────────────────────────
echo -e "\n${GREEN}▶ Opening ${URL}${NC}"
open "$URL" 2>/dev/null || echo -e "  ${AMBER}Open your browser to:${NC} ${URL}"

echo -e "\n${DIM}Dashboard is running. Keep this window open; close it (or Ctrl-C) to stop.${NC}"
# If we started the server in this session, keep the window alive tailing logs.
if [ -n "$SERVER_PID" ]; then
  trap 'echo; echo "Stopping dashboard…"; kill '"$SERVER_PID"' 2>/dev/null; exit 0' INT TERM
  wait "$SERVER_PID"
else
  echo -e "${DIM}(Server was already running in another window.)${NC}"
  echo "Press any key to close this window…"; read -n 1 -s -r
fi
