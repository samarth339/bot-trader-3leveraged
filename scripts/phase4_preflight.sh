#!/usr/bin/env bash
# phase4_preflight.sh — One-time pre-flight check before starting paper trading.
# Run this once to verify every dependency is satisfied.
# Usage: bash scripts/phase4_preflight.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'
PASS=0; FAIL=0; WARN=0

ok()   { echo -e "  ${GREEN}✓${RESET}  $1"; ((PASS++)) || true; }
fail() { echo -e "  ${RED}✗${RESET}  $1"; ((FAIL++)) || true; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; ((WARN++)) || true; }

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Phase 4 Pre-Flight Check${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════${RESET}"
echo ""

# ── 1. Python environment ─────────────────────────────────────────────────────
echo "── Python environment ──────────────────────────"

if python3 --version &>/dev/null; then
  ok "Python 3 available  ($(python3 --version))"
else
  fail "Python 3 not found — install via pyenv or brew"
fi

for pkg in ib_insync pandas_market_calendars pytz yfinance pandas; do
  if python3 -c "import $pkg" &>/dev/null; then
    ok "$pkg installed"
  else
    fail "$pkg not installed — run: pip install -r requirements.txt"
  fi
done

# ── 2. IB Gateway ─────────────────────────────────────────────────────────────
echo ""
echo "── IB Gateway (paper) ──────────────────────────"

if python3 - <<'PYEOF' 2>/dev/null
import socket, sys
try:
    s = socket.create_connection(("127.0.0.1", 4002), timeout=3)
    s.close()
except Exception:
    sys.exit(1)
PYEOF
then
  ok "Port 4002 open — IB Gateway is running"
else
  fail "Port 4002 closed — IB Gateway is NOT running"
  echo ""
  echo -e "     ${YELLOW}To fix:${RESET}"
  echo "     1. Open IB Gateway → log in to paper account"
  echo "     2. Configure → Settings → API → Settings"
  echo "        Enable ActiveX and Socket Clients: ✓"
  echo "        Socket port: 4002"
  echo "        Trusted IP: 127.0.0.1"
  echo "     3. Re-run this script"
  echo ""
fi

# ── 3. Shadow mode ────────────────────────────────────────────────────────────
echo "── Shadow mode ─────────────────────────────────"

if [ -f "logs/shadow_state.json" ]; then
  ok "logs/shadow_state.json exists"
  if python3 -c "
import json
s = json.load(open('logs/shadow_state.json'))
assert s.get('completed'), 'not completed'
print(f\"  day_number={s.get('day_number')}, last_run={s.get('last_run_date')}\")
" 2>/dev/null; then
    SHADOW_INFO=$(python3 -c "
import json
s = json.load(open('logs/shadow_state.json'))
print(f\"day {s.get('day_number')}/30, last run {s.get('last_run_date')}\")
")
    ok "Shadow mode completed  ($SHADOW_INFO)"
  else
    fail "Shadow mode not completed — cannot trade yet"
  fi
else
  fail "logs/shadow_state.json not found"
fi

# ── 4. Signal data ────────────────────────────────────────────────────────────
echo ""
echo "── Signal data ─────────────────────────────────"

if [ -f "logs/signal_history.csv" ]; then
  ok "logs/signal_history.csv exists"
  SIG_RESULT=$(python3 - 2>/dev/null <<'PYEOF'
import pandas as pd, sys
from datetime import date, timedelta
df = pd.read_csv("logs/signal_history.csv")
last_date = pd.to_datetime(df["as_of_date"].iloc[-1]).date()
days_old = (date.today() - last_date).days
regime = df.iloc[-1]["regime"]
alloc_a = int(float(df.iloc[-1]["weight_a"]) * 100)
if days_old <= 1:
    print(f"OK Latest signal: {last_date}  regime={regime}  alloc={alloc_a}% A / {100-alloc_a}% B")
elif days_old <= 3:
    print(f"WARN Signal is {days_old} days old ({last_date}) — run: git pull")
else:
    print(f"FAIL Signal is stale ({days_old} days old, {last_date})")
    sys.exit(1)
PYEOF
) && true
  if [ -z "$SIG_RESULT" ]; then
    warn "Could not parse signal — pandas may not be installed yet"
  elif [[ "$SIG_RESULT" == OK* ]]; then
    ok "${SIG_RESULT#OK }"
  elif [[ "$SIG_RESULT" == WARN* ]]; then
    warn "${SIG_RESULT#WARN }"
  else
    fail "${SIG_RESULT#FAIL }"
  fi
else
  fail "logs/signal_history.csv not found"
fi

# ── 5. Safety ─────────────────────────────────────────────────────────────────
echo ""
echo "── Safety ──────────────────────────────────────"

if [ ! -f "logs/ibkr_kill.flag" ]; then
  ok "Kill switch: OFF"
else
  fail "Kill switch is ACTIVE — see logs/ibkr_kill.flag"
  echo "     To deactivate: python3 -m ibkr.kill_switch deactivate"
fi

if [ -w "logs" ]; then
  ok "logs/ directory writable"
else
  fail "logs/ directory not writable"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
  echo -e "  ${GREEN}${BOLD}All $PASS checks passed — ready for paper trading.${RESET}"
  echo ""
  echo "  Next step:  bash scripts/phase4_dryrun.sh"
else
  echo -e "  ${RED}${BOLD}$FAIL check(s) failed.${RESET} Fix the above before proceeding."
  if [ "$WARN" -gt 0 ]; then
    echo -e "  ${YELLOW}$WARN warning(s).${RESET}"
  fi
fi
echo ""
exit "$FAIL"
